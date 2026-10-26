
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_dataset import SpaceNetDataset
from segformer_model import create_segformer


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_PATH = "dataset/metadata.csv"

IMAGE_SIZE = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

CHECKPOINT_DIR = "checkpoints"

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "segformer_best.pth"
)

HISTORY_PATH = "segformer_history.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "validation_loss": [],
    "validation_iou": [],
    "validation_dice": []
}


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
        (2.0 * intersection + smooth)
        /
        (
            probs.sum(dim=1)
            + targets.sum(dim=1)
            + smooth
        )
    )

    return 1.0 - dice.mean()


# ============================================================
# COMBINED BCE + DICE LOSS
# ============================================================

def combined_loss(logits, targets):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(
        logits,
        targets
    )

    return bce + dice


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6
):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
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
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    iou = (
        (intersection + smooth)
        /
        (union + smooth)
    )

    return iou.mean().item()


# ============================================================
# DICE METRIC
# ============================================================

def calculate_dice(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
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
        2.0 * intersection + 1e-7
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + 1e-7
    )

    return dice.mean().item()


# ============================================================
# RESIZE BATCH
# ============================================================

def resize_batch(images, masks):

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

    return images, masks


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_iou = 0.0

    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        images, masks = resize_batch(
            images,
            masks
        )

        optimizer.zero_grad()

        outputs = model(
            pixel_values=images
        )

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        loss = combined_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        batch_iou = calculate_iou(
            logits,
            masks
        )

        total_loss += loss.item()
        total_iou += batch_iou

        print(
            f"  Batch {batch_idx + 1}/{len(loader)} "
            f"| Loss: {loss.item():.4f} "
            f"| IoU: {batch_iou:.4f}"
        )

    average_loss = (
        total_loss / len(loader)
    )

    average_iou = (
        total_iou / len(loader)
    )

    return average_loss, average_iou


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            images, masks = resize_batch(
                images,
                masks
            )

            outputs = model(
                pixel_values=images
            )

            logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            loss = combined_loss(
                logits,
                masks
            )

            batch_iou = calculate_iou(
                logits,
                masks
            )

            batch_dice = calculate_dice(
                logits,
                masks
            )

            total_loss += loss.item()
            total_iou += batch_iou
            total_dice += batch_dice

    average_loss = (
        total_loss / len(loader)
    )

    average_iou = (
        total_iou / len(loader)
    )

    average_dice = (
        total_dice / len(loader)
    )

    return (
        average_loss,
        average_iou,
        average_dice
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEGFORMER TRAINING")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Epochs:", NUM_EPOCHS)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = SpaceNetDataset(
        metadata_path=METADATA_PATH,
        split="train"
    )

    val_dataset = SpaceNetDataset(
        metadata_path=METADATA_PATH,
        split="validation"
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading SegFormer...")

    model = create_segformer()
    model = model.to(DEVICE)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_iou = -1.0

    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(NUM_EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_iou = train_one_epoch(
            model,
            train_loader,
            optimizer,
            DEVICE
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_iou, val_dice = validate(
            model,
            val_loader,
            DEVICE
        )

        # ----------------------------------------------------
        # STORE HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["validation_loss"].append(
            val_loss
        )

        history["validation_iou"].append(
            val_iou
        )

        history["validation_dice"].append(
            val_dice
        )

        # ----------------------------------------------------
        # PRINT EPOCH RESULTS
        # ----------------------------------------------------

        print("\nEpoch results:")

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train IoU : {train_iou:.4f}"
        )

        print(
            f"Val Loss  : {val_loss:.4f}"
        )

        print(
            f"Val IoU   : {val_iou:.4f}"
        )

        print(
            f"Val Dice  : {val_dice:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_iou > best_val_iou:

            best_val_iou = val_iou

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_loss": val_loss,
                },
                BEST_MODEL_PATH
            )

            print("\nBest model saved!")

            print(
                f"Validation IoU: {val_iou:.4f}"
            )

        print("=" * 70)

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best validation IoU: {best_val_iou:.4f}"
    )

    print(
        f"Best model: {BEST_MODEL_PATH}"
    )

    print(
        f"History: {HISTORY_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
