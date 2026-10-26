import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_dataset import SpaceNetDataset
from deeplabv3plus_resnet50_model import (
    get_model,
    set_batchnorm_eval
)

# =========================
# Configuration
# =========================

METADATA_PATH = "dataset/metadata.csv"

IMAGE_SIZE = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 10

BEST_MODEL_PATH = "deeplabv3plus_resnet50_best.pth"
HISTORY_PATH = "deeplabv3plus_resnet50_history.json"


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================
# Dice Loss
# =========================

def dice_loss(logits, targets, smooth=1.0):

    probs = torch.sigmoid(logits)

    probs = probs.view(-1)#[batch, channel, height, width]
    targets = targets.view(-1)

    intersection = (probs * targets).sum() #how much the prediction overlaps with the actual building mask.
#overlap between your predicted building regions and the actual building regions.
    dice = (
        (2.0 * intersection + smooth)
        / (probs.sum() + targets.sum() + smooth)
    )

    return 1.0 - dice


# =========================
# IoU
# =========================

def calculate_iou(logits, targets, threshold=0.5):

    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    if union == 0:
        return 1.0

    return (intersection / union).item()


# =========================
# Dice Score
# =========================

def calculate_dice(logits, targets, threshold=0.5):

    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()

    intersection = (preds * targets).sum()

    denominator = preds.sum() + targets.sum()

    if denominator == 0:
        return 1.0

    return (
        (2.0 * intersection) / denominator
    ).item()


# =========================
# Dataset
# =========================

train_dataset = SpaceNetDataset(
    METADATA_PATH,
    split="train"
)

val_dataset = SpaceNetDataset(
    METADATA_PATH,
    split="validation"
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# =========================
# Data Loaders
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# Model
# =========================

model = get_model().to(device)
set_batchnorm_eval(model)


print("Model: DeepLabV3+ with ResNet50 encoder")
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# =========================
# Loss and Optimizer
# =========================

bce_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Training History
# =========================

history = {
    "train_loss": [],
    "val_loss": [],
    "val_iou": [],
    "val_dice": []
}


best_val_iou = -1.0


# =========================
# Training Loop
# =========================

for epoch in range(EPOCHS):

    model.train()
    set_batchnorm_eval(model)

    running_loss = 0.0

    for images, masks in train_loader:

        # Use first 3 channels
        images = images[:, :3, :, :]

        # Resize images
        images = F.interpolate(
            images,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        # Resize masks
        masks = F.interpolate(
            masks,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest"
        )

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # BCE + Dice
        loss_bce = bce_loss(outputs, masks)
        loss_dice = dice_loss(outputs, masks)

        loss = loss_bce + loss_dice

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)


    # =========================
    # Validation
    # =========================

    model.eval()

    val_loss_total = 0.0
    val_iou_total = 0.0
    val_dice_total = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images[:, :3, :, :]

            images = F.interpolate(
                images,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            masks = F.interpolate(
                masks,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="nearest"
            )

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss_bce = bce_loss(outputs, masks)
            loss_dice = dice_loss(outputs, masks)

            loss = loss_bce + loss_dice

            val_loss_total += loss.item()

            val_iou_total += calculate_iou(
                outputs,
                masks
            )

            val_dice_total += calculate_dice(
                outputs,
                masks
            )


    val_loss = val_loss_total / len(val_loader)

    val_iou = val_iou_total / len(val_loader)

    val_dice = val_dice_total / len(val_loader)


    # =========================
    # Save History
    # =========================

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)
    history["val_dice"].append(val_dice)


    # =========================
    # Save Best Model
    # =========================

    if val_iou > best_val_iou:

        best_val_iou = val_iou

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("  -> Best model saved")


    # =========================
    # Print Epoch Results
    # =========================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val IoU: {val_iou:.4f} "
        f"Val Dice: {val_dice:.4f}"
    )


# =========================
# Save Training History
# =========================

with open(HISTORY_PATH, "w") as f:

    json.dump(
        history,
        f,
        indent=4
    )


print()
print("Training complete.")
print("Best validation IoU:", best_val_iou)
print("Best model:", BEST_MODEL_PATH)
print("History:", HISTORY_PATH)