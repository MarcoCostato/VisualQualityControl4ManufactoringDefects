import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18
import cv2

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)
    

class ConvTBlock(nn.Module):
    def __init__(self, in_ch, out_ch, last=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() if last else nn.BatchNorm2d(out_ch),
            nn.Identity() if last else nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, bottleneck_dim=512):
        super().__init__()

        self.enc_conv = nn.Sequential(
            ConvBlock(3, 32),   #256 -> 128
            ConvBlock(32, 64),  #128 -> 64
            ConvBlock(64, 128), #64 -> 32
            ConvBlock(128, 256) #32 -> 16
        )
        self.flatten = nn.Flatten()
        self.enc_fc = nn.Linear(16*16*256, bottleneck_dim)

        self.dec_fc = nn.Linear(bottleneck_dim, 16*16*256)
        self.unflatten = nn.Unflatten(1, (256, 16, 16))
        self.dec_conv = nn.Sequential(
            ConvTBlock(256, 128),
            ConvTBlock(128, 64),
            ConvTBlock(64, 32),
            ConvTBlock(32, 3, last=True)
        )

    def forward(self, x):
        z = self.enc_fc(self.flatten(self.enc_conv(x)))
        return self.dec_conv(self.unflatten(self.dec_fc(z)))
    
    def  encode(self, x):
        return self.enc_fc(self.flatten(self.enc_conv(x)))
    
model = resnet18(weights='IMAGENET1K_V1')

feature_extractor = nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2
)


# --- Config ---
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 256
AE_PATH      = "models/autoencoder_leather_b512.pth"
PC_PATH      = "models/coreset_leather.pth"
BOTTLENECK = 512

# --- Transforms ---
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor.cpu() * std + mean).clamp(0, 1)




# --- Load models ---
# Autoencoder
ae_model = AutoEncoder(bottleneck_dim=BOTTLENECK).to(DEVICE)
ae_checkpoint = torch.load(AE_PATH, map_location=DEVICE, weights_only=False)
ae_model.load_state_dict(ae_checkpoint["model_state"])
ae_model.eval()

# PatchCore
coreset = torch.load(PC_PATH, map_location=DEVICE)
extractor     = feature_extractor.to(DEVICE)
extractor.eval()

print("Models loaded.")

# --- Inference functions ---
def run_autoencoder(image_tensor):
    with torch.no_grad():
        recon = ae_model(image_tensor.to(DEVICE))
    error_map = ((image_tensor.to(DEVICE) - recon) ** 2).mean(dim=1)
    error_map = error_map.squeeze().cpu().numpy()
    score     = float(np.percentile(error_map, 95))
    return error_map, score

def run_patchcore(image_tensor):
    with torch.no_grad():
        features = extractor(image_tensor.to(DEVICE))   # [1, 128, 32, 32]
    B, C, H, W = features.shape
    patches    = features.permute(0, 2, 3, 1).reshape(-1, C)
    dists      = torch.cdist(patches, coreset)
    nn_dists   = dists.min(dim=1).values
    anomaly_map = nn_dists.reshape(H, W).cpu()
    score       = anomaly_map.max().item()
    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()
    return anomaly_map, score

def make_heatmap_figure(anomaly_map, score, method):
    """Render anomaly map as a clean matplotlib figure."""
    smoothed = gaussian_filter(anomaly_map, sigma=4)
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(smoothed, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"{method} — anomaly score: {score:.4f}", fontsize=11)
    fig.tight_layout()
    return fig



def get_anomaly_overlay(original_np, error_map, blur_sigma=4, threshold_percentile=97, alpha=0.4):
    """
    original_np : numpy [H, W, 3] in range [0, 1]
    error_map   : numpy [H, W]    raw per-pixel MSE
    Returns     : overlaid image [H, W, 3]
    """
    # 1. Smooth
    smoothed = gaussian_filter(error_map, sigma=blur_sigma)

    # 2. Normalize to [0, 1]
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)

    # 3. Threshold
    threshold = np.percentile(smoothed, threshold_percentile)
    binary_mask = (smoothed > threshold).astype(np.uint8)

    # 4. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
    #binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Colored heatmap (jet colormap over smoothed map)
    heatmap = plt.cm.jet(smoothed)[:, :, :3]   # [H, W, 3], drop alpha channel

    # 6. Blend heatmap only where mask is active
    overlay = original_np.copy()
    mask_bool = binary_mask.astype(bool)          # [H, W]
    overlay[mask_bool] = (
        (1 - alpha) * original_np[mask_bool] +
        alpha       * heatmap[mask_bool]
    )

    # 7. Draw contour around detected region
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    cv2.drawContours(overlay_uint8, contours, -1, (255, 0, 0), 2)   # red contour

    return overlay_uint8, smoothed, binary_mask


# --- Main predict function ---
def predict(pil_image, method):
    if pil_image is None:
        return None, "No image provided."

    # Preprocess
    image_tensor = transform(pil_image).unsqueeze(0)   # [1, 3, 256, 256]

    # Run selected method
    if method == "Autoencoder":
        anomaly_map, score = run_autoencoder(image_tensor)
    else:
        anomaly_map, score = run_patchcore(image_tensor)

    # Verdict
    threshold = 1.104 if method == "Autoencoder" else 2.471  # tune these
    is_anomaly = score > threshold
    verdict   = "🔴 ANOMALY DETECTED" if is_anomaly else "🟢 NORMAL"

    # Heatmap figure
    fig = make_heatmap_figure(anomaly_map, score, method)

    if is_anomaly:
        orig_np = np.array(pil_image.resize((IMG_SIZE,IMG_SIZE))).astype(np.float32) / 255.0
        overlay, _, _ = get_anomaly_overlay(orig_np, anomaly_map)
        overlay_pil = Image.fromarray(overlay)
    else:
        overlay_pil = None

    return fig, overlay_pil, f"{verdict}\nScore: {score:.4f}"

# --- Gradio UI ---
with gr.Blocks(title="MVTec Anomaly Detection") as demo:

    gr.Markdown("## 🔍 Visual Anomaly Detection\nUpload a leather texture image to inspect it for defects.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Input image")
            method      = gr.Radio(
                choices=["Autoencoder", "PatchCore"],
                value="PatchCore",
                label="Detection method"
            )
            run_btn     = gr.Button("Run inspection", variant="primary")

        with gr.Column(scale=1):
            score_out   = gr.Textbox(label="Result", lines=2)
            heatmap_out = gr.Plot(label="Anomaly map")
            overlay_out = gr.Image(label="Defect overlay (if anomaly detected)")
            

    run_btn.click(
        fn=predict,
        inputs=[image_input, method],
        outputs=[heatmap_out, overlay_out, score_out]
    )

    gr.Markdown("**Note:** PatchCore is more precise. Autoencoder is faster on CPU.")

demo.launch(share = True)