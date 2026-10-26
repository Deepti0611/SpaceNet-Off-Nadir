import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(PROJECT_ROOT)

from ml_pipeline.pytorch_dataset import SpaceNetDataset
from models.deeplabv3plus_resnet50.deeplabv3plus_resnet50_model import (
    get_model
)


# =========================
# Configuration
# =========================

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "metadata.csv"
)

IMAGE_SIZE = 512
BATCH_SIZE = 1

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "deeplabv3plus_resnet50_partial",
    "deeplabv3plus_resnet50_partial_best.pth"
)

RESULTS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "deeplabv3plus_resnet50_partial",
    "deeplabv3plus_resnet50_partial_test_results.json"
)


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================
# Metrics
# =========================

def calculate_metrics(logits, targets, threshold=0.5):

    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    intersection = tp
    union = tp + fp + fn

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / denominator

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================
# Dataset
# =========================

test_dataset = SpaceNetDataset(
    METADATA_PATH,
    split="test"
)

print("Test samples:", len(test_dataset))


# =========================
# Data Loader
# =========================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# Model
# =========================

model = get_model().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded:", MODEL_PATH)


# =========================
# Overall Metrics
# =========================

overall_metrics = {
    "iou": [],
    "dice": [],
    "precision": [],
    "recall": [],
    "f1": []
}


# =========================
# Per-Angle Results
# =========================

angle_metrics = {}


# =========================
# Evaluation
# =========================

with torch.no_grad():

    for idx, (images, masks) in enumerate(test_loader):

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

        # Calculate metrics for this image
        metrics = calculate_metrics(
            outputs,
            masks
        )

        # -------------------------
        # Overall
        # -------------------------

        for key in overall_metrics:

            overall_metrics[key].append(
                metrics[key]
            )

        # -------------------------
        # Get angle from metadata
        # -------------------------

        sample_info = test_dataset.metadata.iloc[idx]

        angle = str(sample_info["angle"])

        if angle not in angle_metrics:

            angle_metrics[angle] = {
                "iou": [],
                "dice": [],
                "precision": [],
                "recall": [],
                "f1": []
            }

        for key in angle_metrics[angle]:

            angle_metrics[angle][key].append(
                metrics[key]
            )


# =========================
# Average Overall Metrics
# =========================

overall_results = {}

for key, values in overall_metrics.items():

    overall_results[key] = (
        sum(values) / len(values)
    )


# =========================
# Average Per-Angle Metrics
# =========================

angle_results = {}

for angle, metrics in angle_metrics.items():

    angle_results[angle] = {}

    for key, values in metrics.items():

        angle_results[angle][key] = (
            sum(values) / len(values)
        )


# =========================
# Print Results
# =========================

print()
print("=" * 50)
print("PARTIAL FINE-TUNING")
print("OVERALL TEST RESULTS")
print("=" * 50)

for key, value in overall_results.items():

    print(
        f"{key.upper():10s}: {value:.4f}"
    )


print()
print("=" * 50)
print("PARTIAL FINE-TUNING")
print("PER-ANGLE TEST RESULTS")
print("=" * 50)

for angle in sorted(
    angle_results,
    key=lambda x: float(x)
):

    print()
    print("Angle:", angle)

    for key, value in angle_results[angle].items():

        print(
            f"  {key.upper():10s}: {value:.4f}"
        )


# =========================
# Save Results
# =========================

results = {
    "model": "DeepLabV3+",
    "encoder": "ResNet50",
    "transfer_learning": "Partial fine-tuning",
    "image_size": IMAGE_SIZE,
    "threshold": 0.5,
    "overall": overall_results,
    "per_angle": angle_results
}


with open(RESULTS_PATH, "w") as f:

    json.dump(
        results,
        f,
        indent=4
    )


print()
print("Results saved to:", RESULTS_PATH)