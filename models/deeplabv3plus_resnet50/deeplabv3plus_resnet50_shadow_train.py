import sys
import os

# =========================
# Add project root to path
# =========================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)


import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from ml_pipeline.shadow_augmented_dataset import (
    ShadowAugmentedSpaceNetDataset
)

from ml_pipeline.pytorch_dataset import SpaceNetDataset

from models.deeplabv3plus_resnet50.deeplabv3plus_resnet50_model import (
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

BEST_MODEL_PATH = (
    "models/deeplabv3plus_resnet50/"
    "deeplabv3plus_resnet50_shadow_best.pth"
)

HISTORY_PATH = (
    "models/deeplabv3plus_resnet50/"
    "deeplabv3plus_resnet50_shadow_history.json"
)


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================
# Dice Loss
# =========================

def dice_loss(logits, targets):

    probs = torch.sigmoid(logits)

    smooth = 1e-6

    intersection = (probs * targets).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        probs.sum() + targets.sum() + smooth
    )

    return 1 - dice


# =========================
# IoU
# =========================

def calculate_iou(logits, targets):

    probs = torch.sigmoid(logits)

    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum()

    union = (
        preds.sum()
        + targets.sum()
        - intersection
    )

    if union == 0:
        return 1.0

    return (
        intersection / union
    ).item()


# =========================
# Validation Metrics
# =========================

def calculate_dice(logits, targets):

    probs = torch.sigmoid(logits)

    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum()

    denominator = (
        preds.sum()
        + targets.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2 * intersection / denominator
    ).item()


# =========================
# Dataset
# =========================

train_dataset = ShadowAugmentedSpaceNetDataset(
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

print("Model: DeepLabV3+ with ResNet50 encoder")
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

set_batchnorm_eval(model)


# =========================
# Loss
# =========================

bce_loss = nn.BCEWithLogitsLoss()


# =========================
# Optimizer
# =========================

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


best_val_iou = 0.0


# =========================
# Training Loop
# =========================

for epoch in range(EPOCHS):

    model.train()

    # Keep BatchNorm layers in evaluation mode
    set_batchnorm_eval(model)

    train_loss = 0.0

    for images, masks in train_loader:

        # Use RGB channels
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

        # Loss
        loss_bce = bce_loss(
            outputs,
            masks
        )

        loss_dice = dice_loss(
            outputs,
            masks
        )

        loss = loss_bce + loss_dice

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    # =========================
    # Validation
    # =========================

    model.eval()

    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0

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

            loss_bce = bce_loss(
                outputs,
                masks
            )

            loss_dice = dice_loss(
                outputs,
                masks
            )

            loss = loss_bce + loss_dice

            val_loss += loss.item()

            val_iou += calculate_iou(
                outputs,
                masks
            )

            val_dice += calculate_dice(
                outputs,
                masks
            )


    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)


    # =========================
    # Save History
    # =========================

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_iou"].append(
        val_iou
    )

    history["val_dice"].append(
        val_dice
    )


    # =========================
    # Print Results
    # =========================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val IoU: {val_iou:.4f} "
        f"Val Dice: {val_dice:.4f}"
    )


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
# Save History
# =========================

with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# =========================
# Training Complete
# =========================

print()
print("Training complete.")

print(
    "Best validation IoU:",
    best_val_iou
)

print(
    "Best model:",
    BEST_MODEL_PATH
)

print(
    "History:",
    HISTORY_PATH
)