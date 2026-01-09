# ------------------------------------------------------------
# U-Net Image Segmentation (Improved Version)
# ------------------------------------------------------------
import os
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# -----------------------
# Parameters
# -----------------------
EPOCHS = 150           # ⬆️ Increased for better convergence
IMG_SIZE = 128
BATCH_SIZE = 8
LR = 1e-3

# Paths
TRAIN_PATH = r"/Users/pratikborle/Desktop/trial/data/images"
MASK_PATH = r"/Users/pratikborle/Desktop/trial/data/masks"

print("Images found:", len(os.listdir(TRAIN_PATH)))
print("Masks found:", len(os.listdir(MASK_PATH)))

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -----------------------
# Dataset Class (with Augmentation + Normalization)
# -----------------------
class SaltDataset(Dataset):
    def __init__(self, image_ids, images_path, masks_path, augment=False):
        self.image_ids = image_ids
        self.images_path = images_path
        self.masks_path = masks_path
        self.augment = augment

    def __len__(self):
        return len(self.image_ids)

    def random_transform(self, img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            angle = random.randint(-15, 15)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        return img, mask

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_path, img_id)
        mask_path = os.path.join(self.masks_path, img_id)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Error reading {img_path} or {mask_path}")

        # Resize and normalize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # Convert to tensor-like format
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Normalize to [-1,1]
        img = (img - 0.5) / 0.5

        # Random augmentation
        if self.augment:
            img, mask = self.random_transform(img, mask)

        return img, mask


# -----------------------
# Split Dataset
# -----------------------
valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
all_ids = [f for f in os.listdir(TRAIN_PATH) if f.lower().endswith(valid_exts)]
train_ids, val_ids = train_test_split(all_ids, test_size=1, random_state=42)

train_dataset = SaltDataset(train_ids, TRAIN_PATH, MASK_PATH, augment=True)
val_dataset = SaltDataset(val_ids, TRAIN_PATH, MASK_PATH, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# U-Net Model with Dropout
# -----------------------
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
        self.dconv_down1 = DoubleConv(in_ch, 64, dropout=0.1)
        self.dconv_down2 = DoubleConv(64, 128, dropout=0.1)
        self.dconv_down3 = DoubleConv(128, 256, dropout=0.2)
        self.dconv_down4 = DoubleConv(256, 512, dropout=0.3)

        self.maxpool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dconv_up3 = DoubleConv(512, 256, dropout=0.2)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dconv_up2 = DoubleConv(256, 128, dropout=0.1)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv_up1 = DoubleConv(128, 64, dropout=0.1)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.dconv_down1(x)
        c2 = self.dconv_down2(self.maxpool(c1))
        c3 = self.dconv_down3(self.maxpool(c2))
        c4 = self.dconv_down4(self.maxpool(c3))

        x = self.up3(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.dconv_up3(x)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.dconv_up2(x)

        x = self.up1(x)
        x = torch.cat([x, c1], dim=1)
        x = self.dconv_up1(x)

        return self.final(x)


# -----------------------
# Loss Functions
# -----------------------
bce_loss = nn.BCEWithLogitsLoss()

def dice_coeff(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean()

def focal_loss(pred, target, alpha=0.8, gamma=2):
    pred = torch.sigmoid(pred)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

def combo_loss(pred, target):
    bce = bce_loss(pred, target)
    dice = 1 - dice_coeff(pred, target)
    focal = focal_loss(pred, target)
    return 0.4*bce + 0.4*dice + 0.2*focal


# -----------------------
# Training Setup
# -----------------------
model = UNet(in_ch=1, out_ch=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
best_val_dice = 0

print("\nStarting training...\n")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = combo_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_dice = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            val_dice += dice_coeff(outputs, masks).item()
    val_dice /= len(val_loader)

    scheduler.step(val_dice)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "best_unet_salt.pth")

print("\n✅ Training complete. Best validation Dice coefficient:", best_val_dice)

# -----------------------
# Visualize Predictions in a Grid
# -----------------------
import math

model.eval()
all_imgs, all_masks, all_preds = [], [], []

# Collect all images first
for imgs, masks in val_loader:
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        preds = torch.sigmoid(model(imgs)).cpu().numpy()
    imgs = imgs.cpu().numpy()
    masks = masks.numpy()
    all_imgs.extend(imgs)
    all_masks.extend(masks)
    all_preds.extend(preds)

# Display all predictions as a grid
num_samples = len(all_imgs)
cols = 3
rows = math.ceil(num_samples)
print(f"Displaying {num_samples} validation images...")

for i in range(num_samples):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow((all_imgs[i].squeeze() * 0.5 + 0.5), cmap='gray')
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(all_masks[i].squeeze(), cmap='gray')
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(all_preds[i].squeeze() > 0.5, cmap='gray')
    plt.title("Prediction")

    plt.tight_layout()
    plt.show()
