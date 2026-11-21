
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import SegformerForSemanticSegmentation

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.pytorch_dataset import SpaceNetDataset


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_PATH = "dataset/metadata.csv"

IMAGE_SIZE = 512
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = PROJECT_ROOT / "error_analysis"

VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
CSV_DIR = OUTPUT_DIR / "csv"

VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ALL EXPERIMENT CONFIGURATIONS
# ============================================================

MODEL_CONFIGS = [

    {
        "name": "full_finetuning",
        "display_name": "Full Fine-Tuning",
        "checkpoint": "models/segformer/checkpoints/segformer_best.pth",
        "cbam": False
    },

    {
        "name": "linear_probe",
        "display_name": "Linear Probing",
        "checkpoint": (
            "models/segformer_linear_probe/"
            "checkpoints_linear_probe/"
            "segformer_linear_probe_best.pth"
        ),
        "cbam": False
    },

    {
        "name": "partial_finetuning",
        "display_name": "Partial Fine-Tuning",
        "checkpoint": (
            "models/segformer_partial/"
            "checkpoints_partial/"
            "segformer_partial_best.pth"
        ),
        "cbam": False
    },

    {
        "name": "geometric_augmentation",
        "display_name": "Geometric Augmentation",
        "checkpoint": (
            "models/segformer_geometric/"
            "checkpoints/"
            "segformer_geometric_best.pth"
        ),
        "cbam": False
    },

    {
        "name": "perspective_augmentation",
        "display_name": "Perspective Augmentation",
        "checkpoint": (
            "models/segformer_perspective/"
            "checkpoints/"
            "segformer_perspective_best.pth"
        ),
        "cbam": False
    },

    {
        "name": "shadow_augmentation",
        "display_name": "Shadow Augmentation",
        "checkpoint": (
            "models/segformer_shadow/"
            "checkpoints/"
            "segformer_shadow_best.pth"
        ),
        "cbam": False
    },

    {
        "name": "cbam_bce_dice",
        "display_name": "CBAM + BCE + Dice",
        "checkpoint": (
            "models/segformer_cbam_bce_dice/"
            "checkpoints/"
            "segformer_cbam_best.pth"
        ),
        "cbam": True
    },

    {
        "name": "focal_dice",
        "display_name": "Focal Loss + Dice",
        "checkpoint": (
            "models/segformer_focal_dice/"
            "checkpoints/"
            "segformer_focal_best.pth"
        ),
        "cbam": False
    },

    {
        "name": "cbam_focal_dice",
        "display_name": "CBAM + Focal Loss + Dice",
        "checkpoint": (
            "models/segformer_cbam_focal/"
            "checkpoints/"
            "segformer_cbam_focal_best.pth"
        ),
        "cbam": True
    }
]


# ============================================================
# CBAM ARCHITECTURE
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(channels // reduction, 1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


# ============================================================
# MODEL CREATION
# ============================================================

def create_segformer(use_cbam=False):

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    if use_cbam:

        original_classifier = model.decode_head.classifier

        decoder_channels = model.config.decoder_hidden_size

        model.decode_head.classifier = nn.Sequential(
            CBAM(decoder_channels),
            original_classifier
        )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(config):

    checkpoint_path = PROJECT_ROOT / config["checkpoint"]

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print("\nLoading:", config["display_name"])

    model = create_segformer(
        use_cbam=config["cbam"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint["model_state_dict"]

        print("Checkpoint epoch:", checkpoint.get("epoch", "Unknown"))

        print(
            "Validation IoU:",
            checkpoint.get("val_iou", "Unknown")
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model, checkpoint


# ============================================================
# VIEWING ANGLE
# ============================================================

def get_angle(row):

    for column in [
        "angle",
        "viewing_angle",
        "view_angle"
    ]:

        if column in row.index:

            value = str(row[column]).strip()

            if value.lower() not in ["nan", "none", ""]:
                return value

    return "unknown"


def normalize_angle(angle):

    angle = str(angle).strip().lower()

    for value in ["7", "10", "29", "39", "47", "53"]:

        if angle.startswith(value):
            return f"{value}°"

    return angle


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):

    prediction = prediction.bool()
    target = target.bool()

    tp = torch.logical_and(
        prediction, target
    ).sum().item()

    fp = torch.logical_and(
        prediction, ~target
    ).sum().item()

    fn = torch.logical_and(
        ~prediction, target
    ).sum().item()

    tn = torch.logical_and(
        ~prediction, ~target
    ).sum().item()

    return calculate_metrics_from_counts(
        tp, fp, fn, tn
    )


def calculate_metrics_from_counts(tp, fp, fn, tn):

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

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "IoU": iou,
        "Dice": dice,
        "F1": dice,
        "Precision": precision,
        "Recall": recall,
        "FPR": fpr,
        "FNR": fnr
    }


# ============================================================
# ERROR MAP
# ============================================================

def create_error_map(prediction, target):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    height, width = target.shape

    error_map = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    true_positive = prediction & target
    false_positive = prediction & ~target
    false_negative = ~prediction & target

    # Green = True Positive
    error_map[true_positive] = [0, 255, 0]

    # Red = False Positive
    error_map[false_positive] = [255, 0, 0]

    # Blue = False Negative
    error_map[false_negative] = [0, 0, 255]

    # Black = True Negative

    return error_map


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    image,
    ground_truth,
    prediction,
    error_map,
    image_name,
    angle,
    metrics,
    output_folder
):

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    axes[0].imshow(image)
    axes[0].set_title("Original Satellite Image")

    axes[1].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Ground Truth")

    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[2].set_title("Predicted Mask")

    axes[3].imshow(error_map)
    axes[3].set_title(
        "Error Map\nGreen: TP | Red: FP | Blue: FN"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{image_name} | Angle: {angle} | "
        f"IoU: {metrics['IoU']:.4f} | "
        f"Dice: {metrics['Dice']:.4f}",
        fontsize=13
    )

    plt.tight_layout()

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        image_name
    )

    save_path = output_folder / f"{safe_name}_analysis.png"

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# ANALYSE ONE MODEL
# ============================================================

def analyse_model(config, test_dataset):

    model_name = config["name"]

    print("\n" + "=" * 75)
    print("ERROR ANALYSIS:", config["display_name"])
    print("=" * 75)

    model, checkpoint = load_model(config)

    output_folder = VISUALIZATION_DIR / model_name

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    overall_tn = 0

    # --------------------------------------------------------
    # TEST LOOP
    # --------------------------------------------------------

    with torch.no_grad():

        for index in range(len(test_dataset)):

            image, mask = test_dataset[index]

            image = image.unsqueeze(0).to(DEVICE)
            mask = mask.unsqueeze(0).to(DEVICE).float()

            # Resize image
            image = F.interpolate(
                image,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            # Resize mask
            mask = F.interpolate(
                mask,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="nearest"
            )

            # Model prediction
            outputs = model(
                pixel_values=image
            )

            logits = F.interpolate(
                outputs.logits,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            probabilities = torch.sigmoid(logits)

            prediction = probabilities >= THRESHOLD
            target = mask >= 0.5

            # Metrics
            metrics = calculate_metrics(
                prediction,
                target
            )

            overall_tp += metrics["TP"]
            overall_fp += metrics["FP"]
            overall_fn += metrics["FN"]
            overall_tn += metrics["TN"]

            # Metadata
            row = test_dataset.metadata.iloc[index]

            image_path = str(row["image_path"])
            image_name = Path(image_path).stem

            angle = normalize_angle(
                get_angle(row)
            )

            # Convert tensors to NumPy
            image_np = (
                image[0]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )

            ground_truth_np = (
                target[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            prediction_np = (
                prediction[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # Error map
            error_map = create_error_map(
                prediction_np,
                ground_truth_np
            )

            # Save four-panel visualization
            save_visualization(
                image=image_np,
                ground_truth=ground_truth_np,
                prediction=prediction_np,
                error_map=error_map,
                image_name=image_name,
                angle=angle,
                metrics=metrics,
                output_folder=output_folder
            )

            # Store per-image metrics
            all_results.append({
                "Image": image_name,
                "Image_Path": image_path,
                "Viewing_Angle": angle,
                **metrics
            })

            print(
                f"{index + 1}/{len(test_dataset)}"
                f" | {image_name}"
                f" | Angle: {angle}"
                f" | IoU: {metrics['IoU']:.4f}"
                f" | Dice: {metrics['Dice']:.4f}"
            )

    # --------------------------------------------------------
    # PER-IMAGE CSV
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="IoU",
        ascending=True
    ).reset_index(drop=True)

    per_image_path = (
        CSV_DIR / f"{model_name}_per_image.csv"
    )

    results_df.to_csv(
        per_image_path,
        index=False
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    overall_metrics = calculate_metrics_from_counts(
        overall_tp,
        overall_fp,
        overall_fn,
        overall_tn
    )

    overall_result = {
        "Model": config["display_name"],
        "Checkpoint": config["checkpoint"],
        "Checkpoint_Epoch": (
            checkpoint.get("epoch")
            if isinstance(checkpoint, dict)
            else None
        ),
        "Test_Samples": len(test_dataset),
        **overall_metrics
    }

    # --------------------------------------------------------
    # PER-ANGLE METRICS
    # --------------------------------------------------------

    angle_results = []

    for angle, group in results_df.groupby("Viewing_Angle"):

        angle_metrics = calculate_metrics_from_counts(
            group["TP"].sum(),
            group["FP"].sum(),
            group["FN"].sum(),
            group["TN"].sum()
        )

        angle_results.append({
            "Model": config["display_name"],
            "Viewing_Angle": angle,
            "Images": len(group),
            **angle_metrics
        })

    angle_df = pd.DataFrame(angle_results)

    angle_path = (
        CSV_DIR / f"{model_name}_per_angle.csv"
    )

    angle_df.to_csv(
        angle_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE OVERALL CSV
    # --------------------------------------------------------

    overall_path = (
        CSV_DIR / f"{model_name}_overall.csv"
    )

    pd.DataFrame([overall_result]).to_csv(
        overall_path,
        index=False
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\nOverall Metrics:")

    for metric, value in overall_metrics.items():

        if isinstance(value, float):
            print(f"{metric:<15}: {value:.4f}")
        else:
            print(f"{metric:<15}: {value}")

    print("\nWorst-performing images:")

    print(
        results_df[
            ["Image", "Viewing_Angle", "IoU", "Dice"]
        ].head(5).to_string(index=False)
    )

    print("\nCompleted:", config["display_name"])

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return overall_result, angle_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("SEGFORMER - COMPLETE ERROR ANALYSIS")
    print("=" * 75)

    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Threshold:", THRESHOLD)
    print("Number of models:", len(MODEL_CONFIGS))

    # --------------------------------------------------------
    # Load test dataset ONCE
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_dataset = SpaceNetDataset(
        metadata_path=METADATA_PATH,
        split="test"
    )

    print("Test samples:", len(test_dataset))

    if len(test_dataset) == 0:
        raise RuntimeError("No test samples found.")

    all_overall_results = []
    all_angle_results = []

    failed_models = []

    # --------------------------------------------------------
    # Analyse all models
    # --------------------------------------------------------

    for config in MODEL_CONFIGS:

        try:

            overall_result, angle_results = analyse_model(
                config,
                test_dataset
            )

            all_overall_results.append(overall_result)
            all_angle_results.extend(angle_results)

        except Exception as error:

            print("\nERROR in:", config["display_name"])
            print(error)

            failed_models.append({
                "Model": config["display_name"],
                "Checkpoint": config["checkpoint"],
                "Error": str(error)
            })

            print("Continuing with the next model...")

    # --------------------------------------------------------
    # COMBINED OVERALL RESULTS
    # --------------------------------------------------------

    if all_overall_results:

        overall_df = pd.DataFrame(
            all_overall_results
        )

        overall_df.to_csv(
            CSV_DIR / "all_models_overall_comparison.csv",
            index=False
        )

    # --------------------------------------------------------
    # COMBINED PER-ANGLE RESULTS
    # --------------------------------------------------------

    if all_angle_results:

        angle_df = pd.DataFrame(
            all_angle_results
        )

        angle_df.to_csv(
            CSV_DIR / "all_models_per_angle_comparison.csv",
            index=False
        )

    # --------------------------------------------------------
    # FAILED MODEL REPORT
    # --------------------------------------------------------

    if failed_models:

        pd.DataFrame(failed_models).to_csv(
            CSV_DIR / "failed_models.csv",
            index=False
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("COMPLETE ERROR ANALYSIS FINISHED")
    print("=" * 75)

    print(
        "Successfully analysed:",
        len(all_overall_results)
    )

    print(
        "Failed models:",
        len(failed_models)
    )

    print("\nOverall comparison:")

    if all_overall_results:

        summary_df = pd.DataFrame(
            all_overall_results
        )

        print(
            summary_df[
                ["Model", "IoU", "Dice", "Precision", "Recall"]
            ].to_string(index=False)
        )

    print("\nVisualizations:", VISUALIZATION_DIR)
    print("CSV files:", CSV_DIR)

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
