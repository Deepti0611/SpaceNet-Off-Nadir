import os
import sys
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, ROOT_DIR)

CBAM_MODEL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../deeplabv3plus_cbam_bce_dice"
    )
)

sys.path.insert(0, CBAM_MODEL_DIR)

from pytorch_dataset import SpaceNetDataset
from deeplabv3plus_cbam_model import get_model


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

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "deeplabv3plus_cbam_focal_best.pth"
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "deeplabv3plus_cbam_focal_test_results.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    true_positive = (
        (predictions == 1)
        & (targets == 1)
    ).sum().item()

    false_positive = (
        (predictions == 1)
        & (targets == 0)
    ).sum().item()

    false_negative = (
        (predictions == 0)
        & (targets == 1)
    ).sum().item()

    iou = (
        true_positive
        / (
            true_positive
            + false_positive
            + false_negative
            + 1e-8
        )
    )

    dice = (
        2 * true_positive
        / (
            2 * true_positive
            + false_positive
            + false_negative
            + 1e-8
        )
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
            + 1e-8
        )
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
            + 1e-8
        )
    )

    f1 = (
        2 * precision * recall
        / (
            precision
            + recall
            + 1e-8
        )
    )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    test_dataset = SpaceNetDataset(
        METADATA_PATH,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = get_model().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # The Focal/Dice training script saves a checkpoint
    # dictionary rather than only the state_dict.
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            f"Loaded best model from epoch "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Best validation IoU: "
            f"{checkpoint['val_iou']:.4f}"
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    all_metrics = []

    angle_metrics = {}

    # ========================================================
    # TESTING
    # ========================================================

    with torch.no_grad():

        for idx, (images, masks) in enumerate(
            test_loader
        ):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()

            # Use RGB channels
            images = images[:, :3, :, :]

            # Resize image
            images = F.interpolate(
                images,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            # Resize mask
            masks = F.interpolate(
                masks,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="nearest"
            )

            # Forward pass
            outputs = model(images)

            metrics = calculate_metrics(
                outputs,
                masks
            )

            all_metrics.append(metrics)

            # ------------------------------------------------
            # ANGLE
            # ------------------------------------------------

            metadata_index = idx

            angle = test_dataset.metadata.iloc[
                metadata_index
            ]["angle"]

            angle = str(angle)

            if angle not in angle_metrics:
                angle_metrics[angle] = []

            angle_metrics[angle].append(
                metrics
            )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    overall_results = {}

    metric_names = [
        "iou",
        "dice",
        "precision",
        "recall",
        "f1"
    ]

    for metric in metric_names:

        overall_results[metric] = sum(
            item[metric]
            for item in all_metrics
        ) / len(all_metrics)

    # ========================================================
    # PER-ANGLE RESULTS
    # ========================================================

    per_angle_results = {}

    for angle, metrics_list in angle_metrics.items():

        per_angle_results[angle] = {}

        for metric in metric_names:

            per_angle_results[angle][metric] = (
                sum(
                    item[metric]
                    for item in metrics_list
                )
                / len(metrics_list)
            )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)

    for metric in metric_names:

        print(
            f"{metric.upper():10s}: "
            f"{overall_results[metric]:.4f}"
        )

    print()
    print("=" * 50)
    print("PER-ANGLE RESULTS")
    print("=" * 50)

    for angle in sorted(
        per_angle_results.keys(),
        key=lambda x: float(x)
    ):

        print()
        print(f"Angle {angle}:")

        for metric in metric_names:

            print(
                f"  {metric.upper():10s}: "
                f"{per_angle_results[angle][metric]:.4f}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "model": "DeepLabV3+ ResNet50 + CBAM",
        "loss": "Focal + Dice",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "overall": overall_results,
        "per_angle": per_angle_results
    }

    with open(
        RESULTS_PATH,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print()
    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()