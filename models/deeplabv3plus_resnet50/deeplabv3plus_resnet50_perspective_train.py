
import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml_pipeline.perspective_augmented_dataset import (
    PerspectiveAugmentedSpaceNetDataset
)

from ml_pipeline.pytorch_dataset import SpaceNetDataset

from models.deeplabv3plus_resnet50.deeplabv3plus_resnet50_model import (
    get_model,
    set_batchnorm_eval
)


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_PATH = "dataset/metadata.csv"

IMAGE_SIZE = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 10

BEST_MODEL_PATH = (
    "models/deeplabv3plus_resnet50/"
    "deeplabv3plus_resnet50_perspective_best.pth"
)

HISTORY_PATH = (
    "models/deeplabv3plus_resnet50/"
    "deeplabv3plus_resnet50_perspective_history.json"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# DATASETS
# ============================================================

train_dataset = PerspectiveAugmentedSpaceNetDataset(
    METADATA_PATH,
    split="train"
)

val_dataset = SpaceNetDataset(
    METADATA_PATH,
    split="validation"
)

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

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

model = get_model().to(device)

print("Model: DeepLabV3+ with ResNet50 encoder")
print("Parameters:", sum(p.numel() for p in model.parameters()))

set_batchnorm_eval(model)


# ============================================================
# LOSS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    smooth = 1.0

    intersection = (pred * target).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        pred.sum() + target.sum() + smooth
    )

    return 1.0 - dice


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "val_iou": [],
    "val_dice": []
}

best_val_iou = 0.0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    # Keep BatchNorm layers in evaluation mode
    set_batchnorm_eval(model)

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

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

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = (
        running_loss / len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_total = 0.0

    total_intersection = 0.0
    total_union = 0.0

    total_dice_intersection = 0.0
    total_pred_pixels = 0.0
    total_target_pixels = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

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

            val_loss_total += loss.item()

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities > 0.5
            ).float()

            intersection = (
                predictions * masks
            ).sum().item()

            union = (
                predictions + masks -
                predictions * masks
            ).sum().item()

            total_intersection += intersection
            total_union += union

            total_dice_intersection += intersection
            total_pred_pixels += predictions.sum().item()
            total_target_pixels += masks.sum().item()


    val_loss = (
        val_loss_total / len(val_loader)
    )

    val_iou = (
        total_intersection / total_union
        if total_union > 0
        else 0.0
    )

    val_dice = (
        2.0 * total_dice_intersection
        / (
            total_pred_pixels +
            total_target_pixels
        )
        if (
            total_pred_pixels +
            total_target_pixels
        ) > 0
        else 0.0
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)
    history["val_dice"].append(val_dice)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_iou > best_val_iou:

        best_val_iou = val_iou

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("  -> Best model saved")


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val IoU: {val_iou:.4f} "
        f"Val Dice: {val_dice:.4f}"
    )


# ============================================================
# SAVE HISTORY
# ============================================================

with open(HISTORY_PATH, "w") as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("Training complete.")
print("Best validation IoU:", best_val_iou)
print("Best model:", BEST_MODEL_PATH)
print("History:", HISTORY_PATH)
