import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from pytorch_dataset import SpaceNetDataset


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_PATH = "dataset/metadata.csv"
CHECKPOINT_PATH = "checkpoints/segformer_best.pth"
RESULTS_PATH = "segformer_test_results.json"

IMAGE_SIZE = 512
BATCH_SIZE = 1
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL
# ============================================================

def create_segformer():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, target):
    """
    Calculate pixel-level segmentation metrics.

    pred   : binary prediction tensor
    target : binary ground-truth tensor
    """

    pred = pred.bool()
    target = target.bool()

    tp = torch.logical_and(pred, target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()

    iou_denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    iou = (
        tp / iou_denominator
        if iou_denominator > 0
        else 1.0
    )

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    precision = (
        tp / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        tp / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall
    }


# ============================================================
# DATASET METADATA HELPERS
# ============================================================

def get_angle_from_sample(dataset, index):
    """
    Try to obtain the viewing angle from the dataset metadata.

    The exact metadata representation can vary, so this function
    checks common attribute/column arrangements.
    """

    # Most likely case: dataset has a metadata DataFrame.
    if hasattr(dataset, "metadata"):
        metadata = dataset.metadata

        try:
            row = metadata.iloc[index]

            for column in ["angle", "viewing_angle", "view_angle"]:
                if column in metadata.columns:
                    return str(row[column])
        except Exception:
            pass

    # Alternative naming used by some dataset implementations.
    if hasattr(dataset, "df"):
        metadata = dataset.df

        try:
            row = metadata.iloc[index]

            for column in ["angle", "viewing_angle", "view_angle"]:
                if column in metadata.columns:
                    return str(row[column])
        except Exception:
            pass

    # Last resort: inspect the sample itself if it is a dictionary.
    try:
        sample = dataset.samples[index]

        if isinstance(sample, dict):
            for key in ["angle", "viewing_angle", "view_angle"]:
                if key in sample:
                    return str(sample[key])
    except Exception:
        pass

    return "unknown"


def normalize_angle(angle):
    """
    Convert different possible angle representations to a standard label.
    """

    angle = str(angle).strip().lower()

    # Handle values such as 7, 7.0, "7°", "7 degrees"
    for value in ["7", "10", "29", "39", "47", "53"]:
        if angle.startswith(value):
            return f"{value}°"

    return str(angle)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("SEGFORMER TEST SET EVALUATION")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Image size:", IMAGE_SIZE)
    print("Threshold:", THRESHOLD)

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_dataset = SpaceNetDataset(
        metadata_path=METADATA_PATH,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Test samples:", len(test_dataset))

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading SegFormer...")

    model = create_segformer()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # Training script saved a dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint.get("epoch", "unknown")
        )

        print(
            "Checkpoint validation IoU:",
            checkpoint.get("val_iou", "unknown")
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:
        # Fallback if only a state_dict was saved.
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Overall counters
    # --------------------------------------------------------

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    # --------------------------------------------------------
    # Per-angle counters
    # --------------------------------------------------------

    angles = ["7°", "10°", "29°", "39°", "47°", "53°"]

    angle_stats = {
        angle: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "images": 0
        }
        for angle in angles
    }

    unknown_angle_stats = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "images": 0
    }

    # --------------------------------------------------------
    # Test loop
    # --------------------------------------------------------

    print("\nStarting evaluation...")
    print("-" * 70)

    with torch.no_grad():

        for batch_idx, (images, masks) in enumerate(test_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()

            # ------------------------------------------------
            # Resize image to 512x512
            # ------------------------------------------------

            images = F.interpolate(
                images,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            # ------------------------------------------------
            # Resize mask using nearest-neighbor
            # ------------------------------------------------

            masks = F.interpolate(
                masks,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="nearest"
            )

            # ------------------------------------------------
            # SegFormer forward pass
            # ------------------------------------------------

            outputs = model(
                pixel_values=images
            )

            logits = outputs.logits

            # SegFormer B0 produces a lower-resolution output.
            logits = F.interpolate(
                logits,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= THRESHOLD
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            batch_metrics = calculate_metrics(
                predictions,
                masks >= 0.5
            )

            overall_tp += batch_metrics["tp"]
            overall_fp += batch_metrics["fp"]
            overall_fn += batch_metrics["fn"]

            # ------------------------------------------------
            # Determine viewing angle
            # ------------------------------------------------

            dataset_index = batch_idx

            angle = get_angle_from_sample(
                test_dataset,
                dataset_index
            )

            angle = normalize_angle(angle)

            if angle in angle_stats:

                angle_stats[angle]["tp"] += batch_metrics["tp"]
                angle_stats[angle]["fp"] += batch_metrics["fp"]
                angle_stats[angle]["fn"] += batch_metrics["fn"]
                angle_stats[angle]["images"] += 1

            else:

                unknown_angle_stats["tp"] += batch_metrics["tp"]
                unknown_angle_stats["fp"] += batch_metrics["fp"]
                unknown_angle_stats["fn"] += batch_metrics["fn"]
                unknown_angle_stats["images"] += 1

            print(
                f"Image {batch_idx + 1}/{len(test_dataset)}"
                f" | Angle: {angle}"
                f" | IoU: {batch_metrics['iou']:.4f}"
                f" | Dice: {batch_metrics['dice']:.4f}"
            )

    # ========================================================
    # FINAL METRIC CALCULATION
    # ========================================================

    def metrics_from_counts(tp, fp, fn):

        iou_denominator = tp + fp + fn
        dice_denominator = 2 * tp + fp + fn
        precision_denominator = tp + fp
        recall_denominator = tp + fn

        iou = (
            tp / iou_denominator
            if iou_denominator > 0
            else 1.0
        )

        dice = (
            2 * tp / dice_denominator
            if dice_denominator > 0
            else 1.0
        )

        precision = (
            tp / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            tp / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        # Dice and F1 are mathematically equivalent for binary
        # segmentation when calculated from the same TP/FP/FN.
        f1 = dice

        return {
            "iou": iou,
            "dice": dice,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_metrics = metrics_from_counts(
        overall_tp,
        overall_fp,
        overall_fn
    )

    # --------------------------------------------------------
    # Per angle
    # --------------------------------------------------------

    per_angle_metrics = {}

    for angle in angles:

        stats = angle_stats[angle]

        if stats["images"] > 0:

            per_angle_metrics[angle] = metrics_from_counts(
                stats["tp"],
                stats["fp"],
                stats["fn"]
            )

            per_angle_metrics[angle]["images"] = stats["images"]

        else:

            per_angle_metrics[angle] = {
                "images": 0,
                "iou": None,
                "dice": None,
                "f1": None,
                "precision": None,
                "recall": None
            }

    # --------------------------------------------------------
    # Results dictionary
    # --------------------------------------------------------

    results = {
        "model": "SegFormer-B0",
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": checkpoint.get(
            "epoch", None
        ) if isinstance(checkpoint, dict) else None,
        "checkpoint_validation_iou": checkpoint.get(
            "val_iou", None
        ) if isinstance(checkpoint, dict) else None,
        "test_samples": len(test_dataset),
        "image_size": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "overall": overall_metrics,
        "per_angle": per_angle_metrics
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SEGFORMER TEST RESULTS")
    print("=" * 70)

    print("\nOverall:")
    print(
        f"  IoU       : {overall_metrics['iou']:.4f}"
    )
    print(
        f"  Dice / F1 : {overall_metrics['dice']:.4f}"
    )
    print(
        f"  Precision : {overall_metrics['precision']:.4f}"
    )
    print(
        f"  Recall    : {overall_metrics['recall']:.4f}"
    )

    print("\nPer viewing angle:")
    print(
        f"{'Angle':<10}"
        f"{'IoU':>10}"
        f"{'Dice/F1':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
    )

    print("-" * 60)

    for angle in angles:

        metrics = per_angle_metrics[angle]

        if metrics["images"] > 0:

            print(
                f"{angle:<10}"
                f"{metrics['iou']:>10.4f}"
                f"{metrics['dice']:>12.4f}"
                f"{metrics['precision']:>12.4f}"
                f"{metrics['recall']:>12.4f}"
            )

    print("-" * 60)

    print(
        f"{'Overall':<10}"
        f"{overall_metrics['iou']:>10.4f}"
        f"{overall_metrics['dice']:>12.4f}"
        f"{overall_metrics['precision']:>12.4f}"
        f"{overall_metrics['recall']:>12.4f}"
    )

    print("\nResults saved to:")
    print(RESULTS_PATH)

    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main() 