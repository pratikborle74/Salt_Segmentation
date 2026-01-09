# ------------------------------------------------------------
# Predict & Display All Images One-by-One (auto next + count)
# ------------------------------------------------------------
import os, cv2, torch, numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 128
MODEL_PATH = r"/Users/pratikborle/Desktop/trial/model/best_unet_salt.pth"
DATA_PATH = r"/Users/pratikborle/Desktop/trial/data/images"
MASK_PATH = r"/Users/pratikborle/Desktop/trial/data/masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)

# -----------------------
# U-Net (same as training)
# -----------------------
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): 
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64, 0.1)
        self.dconv_down2 = DoubleConv(64, 128, 0.1)
        self.dconv_down3 = DoubleConv(128, 256, 0.2)
        self.dconv_down4 = DoubleConv(256, 512, 0.3)
        self.maxpool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dconv_up3 = DoubleConv(512, 256, 0.2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dconv_up2 = DoubleConv(256, 128, 0.1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dconv_up1 = DoubleConv(128, 64, 0.1)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.dconv_down1(x)
        c2 = self.dconv_down2(self.maxpool(c1))
        c3 = self.dconv_down3(self.maxpool(c2))
        c4 = self.dconv_down4(self.maxpool(c3))
        x = self.up3(c4); x = torch.cat([x, c3], 1); x = self.dconv_up3(x)
        x = self.up2(x); x = torch.cat([x, c2], 1); x = self.dconv_up2(x)
        x = self.up1(x); x = torch.cat([x, c1], 1); x = self.dconv_up1(x)
        return self.final(x)


# -----------------------
# Load trained model
# -----------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model Loaded\n")

# -----------------------
# Read images
# -----------------------
valid_ext = (".png", ".jpg", ".jpeg", ".bmp")
files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(valid_ext)]

total = len(files)
print(f"📂 Total images found: {total}")
print("🔎 Starting display...\n")

# -----------------------
# Predict and show each image
# -----------------------
for idx, f in enumerate(files, start=1):
    print(f"▶️ Showing image {idx}/{total}: {f}")

    img_path = os.path.join(DATA_PATH, f)
    mask_path = os.path.join(MASK_PATH, f)

    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"⚠️ Skipping unreadable file: {f}")
        continue

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    t = torch.tensor(img_resized).float().unsqueeze(0).unsqueeze(0)
    t = (t - 0.5) / 0.5
    t = t.to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(t)).cpu().numpy()[0,0]
    pred_mask = (pred > 0.5).astype(np.uint8)

    mask = cv2.imread(mask_path, 0)
    if mask is not None:
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    else:
        mask = np.zeros_like(pred_mask)

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("Input"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(mask, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(pred_mask, cmap="gray"); plt.title("Prediction"); plt.axis("off")
    plt.suptitle(f"{f} ({idx}/{total})")
    plt.show(block=False)
    
    plt.pause(1)  # change duration here if needed (seconds)
    plt.close()

print("\n✅ Completed showing all images!")
