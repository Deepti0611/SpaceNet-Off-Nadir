
import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

MODEL_PATH = (
    "models/deeplabv3plus_resnet50/"
    "deeplabv3plus_resnet50_aug_best.pth"
)

RESULTS_PATH = (
    "models/deeplabv3plus_resnet50/"
    "deeplabv3plus_resnet50_aug_test_results.json"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = SpaceNetDataset(
    METADATA_PATH,
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test samples:", len(test_dataset))


# ============================================================
# LOAD MODEL
# ============================================================

model = get_model().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

set_batchnorm_eval(model)

model.eval()

print("Model: DeepLabV3+ with ResNet50 encoder")
print("Loaded checkpoint:", MODEL_PATH)


# ============================================================
# METRIC STORAGE
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0

angle_metrics = {}


# ============================================================
# TEST
# ============================================================

with torch.no_grad():

    for idx, (images, masks) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        # Resize exactly as done during training
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

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities > 0.5).float()

        # ----------------------------------------------------
        # Pixel-level counts
        # ----------------------------------------------------

        tp = ((predictions == 1) & (masks == 1)).sum().item()
        fp = ((predictions == 1) & (masks == 0)).sum().item()
        fn = ((predictions == 0) & (masks == 1)).sum().item()
        tn = ((predictions == 0) & (masks == 0)).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        # ----------------------------------------------------
        # Get viewing angle from metadata
        # ----------------------------------------------------

        row = test_dataset.metadata.iloc[idx]

        angle = float(row["angle"])

        if angle not in angle_metrics:
            angle_metrics[angle] = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0
            }

        angle_metrics[angle]["tp"] += tp
        angle_metrics[angle]["fp"] += fp
        angle_metrics[angle]["fn"] += fn
        angle_metrics[angle]["tn"] += tn


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(tp, fp, fn, tn):

    iou_denominator = tp + fp + fn

    dice_denominator = (
        2 * tp + fp + fn
    )

    precision_denominator = tp + fp

    recall_denominator = tp + fn

    iou = (
        tp / iou_denominator
        if iou_denominator > 0
        else 0
    )

    dice = (
        (2 * tp) / dice_denominator
        if dice_denominator > 0
        else 0
    )

    precision = (
        tp / precision_denominator
        if precision_denominator > 0
        else 0
    )

    recall = (
        tp / recall_denominator
        if recall_denominator > 0
        else 0
    )

    f1_denominator = precision + recall

    f1 = (
        2 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0
    )

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# OVERALL METRICS
# ============================================================

overall_metrics = calculate_metrics(
    total_tp,
    total_fp,
    total_fn,
    total_tn
)


# ============================================================
# PER-ANGLE METRICS
# ============================================================

per_angle_metrics = {}

for angle in sorted(angle_metrics.keys()):

    counts = angle_metrics[angle]

    per_angle_metrics[str(angle)] = calculate_metrics(
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"]
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("DEEPLABV3+ WITH AUGMENTATION - TEST RESULTS")
print("=" * 60)

print()
print("Overall Test Metrics")

for metric, value in overall_metrics.items():
    print(f"{metric}: {value:.4f}")


print()
print("Per-Angle IoU")

for angle, metrics in per_angle_metrics.items():
    print(
        f"{angle}° -> "
        f"IoU: {metrics['IoU']:.4f}, "
        f"Dice: {metrics['Dice']:.4f}, "
        f"Precision: {metrics['Precision']:.4f}, "
        f"Recall: {metrics['Recall']:.4f}, "
        f"F1: {metrics['F1']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "DeepLabV3Plus",
    "encoder": "ResNet50",
    "training_strategy": "Full Fine-Tuning + Geometric Augmentation",
    "checkpoint": MODEL_PATH,
    "image_size": IMAGE_SIZE,
    "test_samples": len(test_dataset),
    "overall": overall_metrics,
    "per_angle": per_angle_metrics
}


with open(RESULTS_PATH, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )


print()
print("Results saved to:")
print(RESULTS_PATH)
