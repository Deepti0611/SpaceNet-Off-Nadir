import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, ROOT_DIR)

# Reuse the already-tested CBAM DeepLabV3+ model
CBAM_MODEL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../deeplabv3plus_cbam_bce_dice"
    )
)

sys.path.insert(0, CBAM_MODEL_DIR)


from pytorch_dataset import SpaceNetDataset
from deeplabv3plus_cbam_model import get_model, set_batchnorm_eval


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_PATH = os.path.join(
    ROOT_DIR,
    "dataset",
    "metadata.csv"
)

IMAGE_SIZE = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

MODEL_DIR = os.path.dirname(__file__)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deeplabv3plus_cbam_focal_best.pth"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "deeplabv3plus_cbam_focal_history.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(logits, targets, smooth=1e-6):

    probs = torch.sigmoid(logits)

    probs = probs.contiguous().view(
        probs.size(0), -1
    )

    targets = targets.contiguous().view(
        targets.size(0), -1
    )

    intersection = (
        probs * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probs.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return 1.0 - dice.mean()


# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0
):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    probabilities = torch.sigmoid(logits)

    pt = (
        probabilities * targets
        + (1 - probabilities) * (1 - targets)
    )

    alpha_t = (
        alpha * targets
        + (1 - alpha) * (1 - targets)
    )

    focal = (
        alpha_t
        * ((1 - pt) ** gamma)
        * bce
    )

    return focal.mean()


# ============================================================
# COMBINED FOCAL + DICE LOSS
# ============================================================

def combined_loss(logits, targets):

    focal = focal_loss(
        logits,
        targets,
        alpha=0.25,
        gamma=2.0
    )

    dice = dice_loss(
        logits,
        targets
    )

    return focal + dice


# ============================================================
# METRICS
# ============================================================

def calculate_iou(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6
):

    probs = torch.sigmoid(logits)

    predictions = (
        probs > threshold
    ).float()

    predictions = predictions.contiguous().view(
        predictions.size(0), -1
    )

    targets = targets.contiguous().view(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean().item()


def calculate_dice(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6
):

    probs = torch.sigmoid(logits)

    predictions = (
        probs > threshold
    ).float()

    predictions = predictions.contiguous().view(
        predictions.size(0), -1
    )

    targets = targets.contiguous().view(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean().item()


# ============================================================
# MAIN
# ============================================================

def main():

    print(f"Device: {DEVICE}")

    print()
    print("Loss: Focal + Dice")
    print("Focal alpha: 0.25")
    print("Focal gamma: 2.0")
    print()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = SpaceNetDataset(
        METADATA_PATH,
        split="train"
    )

    val_dataset = SpaceNetDataset(
        METADATA_PATH,
        split="validation"
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = get_model().to(DEVICE)

    print()
    print(
        "Model: DeepLabV3+ with ResNet50 encoder + CBAM"
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {total_parameters}")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_iou": [],
        "validation_dice": []
    }

    best_val_iou = -1.0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        # Batch size = 1, so keep BatchNorm layers in eval mode
        set_batchnorm_eval(model)

        running_train_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()

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

            optimizer.zero_grad()

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

        train_loss = (
            running_train_loss
            / len(train_loader)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0
        validation_iou = 0.0
        validation_dice = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE).float()

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

                loss = combined_loss(
                    outputs,
                    masks
                )

                validation_loss += loss.item()

                validation_iou += calculate_iou(
                    outputs,
                    masks
                )

                validation_dice += calculate_dice(
                    outputs,
                    masks
                )

        validation_loss /= len(val_loader)
        validation_iou /= len(val_loader)
        validation_dice /= len(val_loader)

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["validation_iou"].append(
            validation_iou
        )

        history["validation_dice"].append(
            validation_dice
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {validation_loss:.4f} "
            f"Val IoU: {validation_iou:.4f} "
            f"Val Dice: {validation_dice:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if validation_iou > best_val_iou:

            best_val_iou = validation_iou

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": validation_iou,
                "val_loss": validation_loss
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH
            )

            print("-> Best model saved")

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    with open(
        HISTORY_PATH,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    print()
    print("Training complete.")
    print(
        f"Best validation IoU: {best_val_iou}"
    )
    print(
        f"Best model: {os.path.basename(BEST_MODEL_PATH)}"
    )
    print(
        f"History: {os.path.basename(HISTORY_PATH)}"
    )


if __name__ == "__main__":
    main()