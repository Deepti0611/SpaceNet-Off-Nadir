import json
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from pytorch_dataset import SpaceNetDataset
from unet_resnet34_model import get_model


# ============================================================
# Configuration
# ============================================================

METADATA_PATH = "dataset/metadata.csv"

IMAGE_SIZE = 512
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 10

BEST_MODEL_PATH = "unet_resnet34_best.pth"
HISTORY_PATH = "unet_resnet34_history.json"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Loss functions
# ============================================================

def dice_loss(logits, targets, smooth=1.0):

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(-1)
    targets = targets.view(-1)

    intersection = (probabilities * targets).sum()

    dice = (
        (2.0 * intersection + smooth)
        / (
            probabilities.sum()
            + targets.sum()
            + smooth
        )
    )

    return 1.0 - dice


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
# Metrics
# ============================================================

def calculate_dice(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (
        predictions * targets
    ).sum()

    dice = (
        2.0 * intersection + 1e-7
    ) / (
        predictions.sum()
        + targets.sum()
        + 1e-7
    )

    return dice.item()


def calculate_iou(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions.sum()
        + targets.sum()
        - intersection
    )

    if union == 0:
        return 1.0

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return iou.item()


# ============================================================
# Data
# ============================================================

train_dataset = SpaceNetDataset(
    METADATA_PATH,
    "train"
)

validation_dataset = SpaceNetDataset(
    METADATA_PATH,
    "validation"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))


# ============================================================
# Model
# ============================================================

model = get_model().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training history
# ============================================================

history = {
    "train_loss": [],
    "validation_loss": [],
    "validation_iou": [],
    "validation_dice": []
}

best_iou = -1.0


# ============================================================
# Training loop
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):

        # First three channels for ImageNet pretrained encoder
        images = images[:, :3, :, :]

        # Resize images
        images = F.interpolate(
            images,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        # Resize masks using nearest-neighbor
        masks = F.interpolate(
            masks,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest"
        )

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss
        / len(train_loader)
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    total_validation_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for images, masks in validation_loader:

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

            loss = combined_loss(
                outputs,
                masks
            )

            iou = calculate_iou(
                outputs,
                masks
            )

            dice = calculate_dice(
                outputs,
                masks
            )

            total_validation_loss += loss.item()
            total_iou += iou
            total_dice += dice

    average_validation_loss = (
        total_validation_loss
        / len(validation_loader)
    )

    average_iou = (
        total_iou
        / len(validation_loader)
    )

    average_dice = (
        total_dice
        / len(validation_loader)
    )


    # ========================================================
    # Store history
    # ========================================================

    history["train_loss"].append(
        average_train_loss
    )

    history["validation_loss"].append(
        average_validation_loss
    )

    history["validation_iou"].append(
        average_iou
    )

    history["validation_dice"].append(
        average_dice
    )


    # ========================================================
    # Print epoch results
    # ========================================================

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss:      {average_train_loss:.4f}")
    print(f"Validation Loss: {average_validation_loss:.4f}")
    print(f"Validation IoU:  {average_iou:.4f}")
    print(f"Validation Dice: {average_dice:.4f}")


    # ========================================================
    # Save best model
    # ========================================================

    if average_iou > best_iou:

        best_iou = average_iou

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("New best model saved.")


    print("=" * 60)


# ============================================================
# Save training history
# ============================================================

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
print("Best validation IoU:", best_iou)
print("Best model:", BEST_MODEL_PATH)
print("History:", HISTORY_PATH)