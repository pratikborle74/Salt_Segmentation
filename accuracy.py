import os, cv2, torch
import numpy as np
from unet_model import UNet

IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "model/best_unet_salt.pth"
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# Load model
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Dice function
def dice_coeff(pred, true, eps=1e-7):
    pred = (pred > 0.5).astype(np.uint8)
    true = (true > 0.5).astype(np.uint8)
    inter = (pred * true).sum()
    return (2 * inter + eps) / (pred.sum() + true.sum() + eps)

# IoU function
def iou_score(pred, true, eps=1e-7):
    pred = (pred > 0.5).astype(np.uint8)
    true = (true > 0.5).astype(np.uint8)
    inter = (pred & true).sum()
    union = (pred | true).sum()
    return (inter + eps) / (union + eps)

dice_scores, iou_scores = [], []
files = os.listdir(IMAGE_DIR)

for f in files:
    img = cv2.imread(os.path.join(IMAGE_DIR, f), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(MASK_DIR, f), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None: continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0

    # normalize same as training
    img = (img - 0.5) / 0.5

    inp = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(inp)).cpu().numpy()[0,0]

    dice_scores.append(dice_coeff(pred, mask))
    iou_scores.append(iou_score(pred, mask))

print(f"✅ Model Evaluation Complete")
print(f"📊 Average Dice Score: {np.mean(dice_scores):.4f}")
print(f"📊 Average IoU Score:  {np.mean(iou_scores):.4f}")
