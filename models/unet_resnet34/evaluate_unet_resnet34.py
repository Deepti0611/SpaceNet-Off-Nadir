import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_dataset import SpaceNetDataset
from unet_resnet34_model import get_model


# =========================
# Configuration
# =========================

METADATA_PATH = "dataset/metadata.csv"
CHECKPOINT_PATH = "unet_resnet34_best.pth"

IMAGE_SIZE = 512
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Metrics
# =========================

def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(prediction), target).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return iou, dice, precision, recall, f1


# =========================
# Load model
# =========================

print("Device:", device)

model = get_model().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Loaded checkpoint:", CHECKPOINT_PATH)


# =========================
# Load test dataset
# =========================

test_dataset = SpaceNetDataset(
    METADATA_PATH,
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# =========================
# Load metadata for angles
# =========================

metadata = pd.read_csv(METADATA_PATH)

test_metadata = metadata[
    metadata["split"].astype(str).str.lower() == "test"
].reset_index(drop=True)

print("Test metadata rows:", len(test_metadata))


# =========================
# Evaluation
# =========================

overall_tp = 0
overall_fp = 0
overall_fn = 0

angle_stats = {}

for angle in sorted(test_metadata["angle"].unique()):
    angle_stats[angle] = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "samples": 0
    }


with torch.no_grad():

    for batch_index, (images, masks) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        # Resize to training resolution
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

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities >= 0.5).float()

        prediction = predictions[0, 0].cpu().numpy()
        target = masks[0, 0].cpu().numpy()

        prediction = prediction.astype(bool)
        target = target.astype(bool)

        tp = np.logical_and(prediction, target).sum()
        fp = np.logical_and(prediction, np.logical_not(target)).sum()
        fn = np.logical_and(np.logical_not(prediction), target).sum()

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        # Get corresponding viewing angle
        angle = test_metadata.iloc[batch_index]["angle"]

        angle_stats[angle]["tp"] += tp
        angle_stats[angle]["fp"] += fp
        angle_stats[angle]["fn"] += fn
        angle_stats[angle]["samples"] += 1


# =========================
# Overall metrics
# =========================

overall_iou = (
    overall_tp / (overall_tp + overall_fp + overall_fn)
    if (overall_tp + overall_fp + overall_fn) > 0
    else 0.0
)

overall_dice = (
    (2 * overall_tp)
    / (2 * overall_tp + overall_fp + overall_fn)
    if (2 * overall_tp + overall_fp + overall_fn) > 0
    else 0.0
)

overall_precision = (
    overall_tp / (overall_tp + overall_fp)
    if (overall_tp + overall_fp) > 0
    else 0.0
)

overall_recall = (
    overall_tp / (overall_tp + overall_fn)
    if (overall_tp + overall_fn) > 0
    else 0.0
)

overall_f1 = (
    2 * overall_precision * overall_recall
    / (overall_precision + overall_recall)
    if (overall_precision + overall_recall) > 0
    else 0.0
)


# =========================
# Print results
# =========================

print("\n" + "=" * 70)
print("U-NET RESNET34 TEST RESULTS")
print("=" * 70)

print(f"Test IoU:       {overall_iou:.4f}")
print(f"Test Dice:      {overall_dice:.4f}")
print(f"Test Precision: {overall_precision:.4f}")
print(f"Test Recall:    {overall_recall:.4f}")
print(f"Test F1:        {overall_f1:.4f}")

print("\n" + "-" * 70)
print("PER-ANGLE RESULTS")
print("-" * 70)

print(
    f"{'Angle':<10}"
    f"{'Samples':<10}"
    f"{'IoU':<12}"
    f"{'Dice':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 70)

angle_results = {}

for angle in sorted(angle_stats.keys()):

    stats = angle_stats[angle]

    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(
        f"{angle:<10}"
        f"{stats['samples']:<10}"
        f"{iou:<12.4f}"
        f"{dice:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )

    angle_results[str(angle)] = {
        "samples": stats["samples"],
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================
# Save results
# =========================

results = {
    "model": "U-Net with ResNet34 encoder",
    "checkpoint": CHECKPOINT_PATH,
    "test_samples": len(test_dataset),
    "overall": {
        "iou": overall_iou,
        "dice": overall_dice,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1
    },
    "per_angle": angle_results
}

with open("unet_resnet34_test_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:")
print("unet_resnet34_test_results.json")