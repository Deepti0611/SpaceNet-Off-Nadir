"""
Generate all SegFormer comparison tables and plots.

Run from the project root or execute this file directly:
    python analysis/segformer/generate_all_comparison_plots.py

Expected project structure:
    results/segformer/segformer_overall_comparison.csv
    results/segformer/advanced_comparisons/
    models/**/segformer*_test_results.json

The six original experiment filenames follow the user's reference script.
The three CBAM/Focal result files are discovered by filename keywords.
Existing result files are read only; all generated files go into
results/segformer/advanced_comparisons/.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results" / "segformer"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = RESULTS_DIR / "advanced_comparisons"

PAIRWISE_DIR = OUTPUT_DIR / "pairwise_comparisons"
OVERALL_DIR = OUTPUT_DIR / "overall_comparisons"

PAIRWISE_FOLDERS = [
    "experiment_1_vs_2",
    "experiment_1_vs_3",
    "experiment_2_vs_4",
    "experiment_3_vs_4",
    "experiment_1_vs_4",
]

OVERALL_FOLDERS = [
    "fine_tuning",
    "augmentation",
    "cbam_focal",
    "all_experiments",
]

ANGLES = [7, 10, 29, 39, 47, 53]
METRICS = ["IoU", "Dice", "Precision", "Recall"]

# These names and filenames are taken from the uploaded reference script.
ORIGINAL_EXPERIMENT_FILES = {
    "Full Fine-Tuning": "segformer_test_results.json",
    "Linear Probing": "segformer_linear_probe_test_results.json",
    "Partial Fine-Tuning": "segformer_partial_test_results.json",
    "Geometric Augmentation": "segformer_geometric_test_results.json",
    "Perspective Augmentation": "segformer_perspective_test_results.json",
    "Shadow Augmentation": "segformer_shadow_test_results.json",
}

# Ablation experiment names and likely result filenames.
# If your files use different names, the keyword-based search below
# will attempt to locate them automatically.
ABLATION_EXPERIMENTS = {
    "CBAM + BCE + Dice": {
        "filename_candidates": [
            "segformer_cbam_test_results.json",
            "segformer_cbam_bce_dice_test_results.json",
            "segformer_cbam_bce+dice_test_results.json",
        ],
        "keywords": ["cbam"],
        "exclude_keywords": ["focal"],
    },
    "Focal + Dice": {
        "filename_candidates": [
            "segformer_focal_test_results.json",
            "segformer_focal_dice_test_results.json",
        ],
        "keywords": ["focal"],
        "exclude_keywords": ["cbam", "cbma"],
    },
    "CBAM + Focal + Dice": {
        "filename_candidates": [
            "segformer_cbam_focal_test_results.json",
            "segformer_cbam_focal_dice_test_results.json",
        ],
        "keywords": ["cbam", "focal"],
        "exclude_keywords": [],
    },
}

# Four ablation experiments used for the pairwise comparisons.
PAIRWISE_EXPERIMENTS = {
    "experiment_1": "Full Fine-Tuning",
    "experiment_2": "CBAM + BCE + Dice",
    "experiment_3": "Focal + Dice",
    "experiment_4": "CBAM + Focal + Dice",
}

GROUPS = {
    "fine_tuning": [
        "Full Fine-Tuning",
        "Linear Probing",
        "Partial Fine-Tuning",
    ],
    "augmentation": [
        "Geometric Augmentation",
        "Perspective Augmentation",
        "Shadow Augmentation",
    ],
    "cbam_focal": [
        "Full Fine-Tuning",
        "CBAM + BCE + Dice",
        "Focal + Dice",
        "CBAM + Focal + Dice",
    ],
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

for folder in PAIRWISE_FOLDERS:
    (PAIRWISE_DIR / folder).mkdir(parents=True, exist_ok=True)

for folder in OVERALL_FOLDERS:
    (OVERALL_DIR / folder).mkdir(parents=True, exist_ok=True)


# ============================================================
# JSON FILE DISCOVERY
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_exact_filename(filename):
    matches = [
        path for path in ROOT_DIR.rglob(filename)
        if OUTPUT_DIR not in path.parents
    ]

    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' anywhere under:\n{ROOT_DIR}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"More than one file named '{filename}' was found:\n"
            + "\n".join(str(path) for path in matches)
            + "\nPlease keep only one matching result file or edit the path "
              "configuration at the top of this script."
        )

    return matches[0]


def is_result_json(path):
    try:
        data = load_json(path)
        return isinstance(data, dict) and "overall" in data and "per_angle" in data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def find_ablation_file(experiment_config):
    # First try the likely exact filenames.
    for filename in experiment_config["filename_candidates"]:
        matches = [
            path for path in ROOT_DIR.rglob(filename)
            if OUTPUT_DIR not in path.parents
        ]
        valid_matches = [path for path in matches if is_result_json(path)]

        if len(valid_matches) == 1:
            return valid_matches[0]
        if len(valid_matches) > 1:
            raise RuntimeError(
                f"Multiple result files found for '{filename}':\n"
                + "\n".join(str(path) for path in valid_matches)
            )

    # Fallback: discover by filename keywords, not by the JSON model label.
    # This also avoids depending on possible spelling mistakes in that label.
    keywords = experiment_config["keywords"]
    excluded = experiment_config["exclude_keywords"]
    candidates = []

    for path in ROOT_DIR.rglob("*.json"):
        if OUTPUT_DIR in path.parents:
            continue

        filename = path.name.lower()
        if not all(word in filename for word in keywords):
            continue
        if any(word in filename for word in excluded):
            continue
        if "test" not in filename or not is_result_json(path):
            continue

        candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not automatically locate a result JSON for this experiment.\n"
            f"Required filename keywords: {keywords}\n"
            f"Searched under: {ROOT_DIR}\n"
            "Edit ABLATION_EXPERIMENTS near the top of the script and add "
            "the exact filename to filename_candidates."
        )

    raise RuntimeError(
        "More than one possible result JSON was found for this experiment:\n"
        + "\n".join(str(path) for path in candidates)
        + "\nEdit ABLATION_EXPERIMENTS and specify the exact filename."
    )


# ============================================================
# LOAD ALL NINE EXPERIMENTS
# ============================================================

experiment_files = {}

for experiment, filename in ORIGINAL_EXPERIMENT_FILES.items():
    experiment_files[experiment] = find_exact_filename(filename)

for experiment, config in ABLATION_EXPERIMENTS.items():
    experiment_files[experiment] = find_ablation_file(config)

results = {}

print("\nLOADING EXPERIMENT RESULTS")
print("=" * 72)

for experiment, file_path in experiment_files.items():
    data = load_json(file_path)

    if "overall" not in data or "per_angle" not in data:
        raise KeyError(
            f"{file_path} must contain both 'overall' and 'per_angle'."
        )

    results[experiment] = data
    print(f"{experiment}: {file_path}")


# ============================================================
# METRIC EXTRACTION
# ============================================================

JSON_METRIC_KEYS = {
    "IoU": "iou",
    "Dice": "dice",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
}


def get_overall_metric(result, metric):
    key = JSON_METRIC_KEYS[metric]
    value = result["overall"].get(key)

    if value is None:
        raise ValueError(f"Overall metric '{key}' is missing or null.")

    return float(value)


def normalize_angle_key(angle_key):
    """Convert keys such as '7°', '7', or '7.0' to integer degrees."""
    match = re.search(r"\d+", str(angle_key))
    return int(match.group()) if match else None


def get_angle_metrics(result, experiment):
    raw = result["per_angle"]
    normalized = {
        normalize_angle_key(key): value
        for key, value in raw.items()
        if normalize_angle_key(key) is not None
    }

    rows = []
    for angle in ANGLES:
        if angle not in normalized:
            raise KeyError(
                f"Viewing angle {angle} is missing from {experiment} JSON."
            )

        metrics = normalized[angle]
        row = {"Angle": angle}

        for metric in ["IoU", "Dice"]:
            key = JSON_METRIC_KEYS[metric]
            value = metrics.get(key)

            if value is None:
                raise ValueError(
                    f"{metric} is missing for {experiment} at {angle} degrees."
                )

            row[metric] = float(value)

        rows.append(row)

    return rows


overall_rows = []
angle_iou_rows = []
angle_dice_rows = []

for experiment, result in results.items():
    row = {"Method": experiment}

    for metric in ["IoU", "Dice", "Precision", "Recall", "F1"]:
        row[metric] = get_overall_metric(result, metric)

    overall_rows.append(row)

    angle_rows = get_angle_metrics(result, experiment)
    for angle_row in angle_rows:
        angle_iou_rows.append({
            "Angle": angle_row["Angle"],
            "Method": experiment,
            "IoU": angle_row["IoU"],
        })
        angle_dice_rows.append({
            "Angle": angle_row["Angle"],
            "Method": experiment,
            "Dice": angle_row["Dice"],
        })


overall_df = pd.DataFrame(overall_rows)
per_angle_iou_long = pd.DataFrame(angle_iou_rows)
per_angle_dice_long = pd.DataFrame(angle_dice_rows)

# Wide-format per-angle tables, matching the layout in the reference script.
per_angle_iou_df = per_angle_iou_long.pivot(
    index="Angle", columns="Method", values="IoU"
).reset_index()

per_angle_dice_df = per_angle_dice_long.pivot(
    index="Angle", columns="Method", values="Dice"
).reset_index()


# ============================================================
# OPTIONAL: MERGE TRAINING METADATA FROM EXISTING COMPARISON CSV
# ============================================================

metadata_path = RESULTS_DIR / "segformer_overall_comparison.csv"

if metadata_path.exists():
    metadata_df = pd.read_csv(metadata_path)

    if "Configuration" in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"Configuration": "Method"})

    metadata_columns = [
        column for column in [
            "Method",
            "Trainable Parameters",
            "Best Epoch",
            "Best Validation IoU",
            "Best Validation Dice",
            "Test Samples",
        ]
        if column in metadata_df.columns
    ]

    if "Method" in metadata_columns:
        metadata_df = metadata_df[metadata_columns].drop_duplicates("Method")
        overall_df = overall_df.merge(metadata_df, on="Method", how="left")


# Save consolidated source tables only in the new output directory.
overall_df.to_csv(OUTPUT_DIR / "all_experiments_overall_metrics.csv", index=False)
per_angle_iou_df.to_csv(OUTPUT_DIR / "all_experiments_per_angle_iou.csv", index=False)
per_angle_dice_df.to_csv(OUTPUT_DIR / "all_experiments_per_angle_dice.csv", index=False)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def save_bar_comparison(data, methods, metric, output_path, title=None):
    subset = data[data["Method"].isin(methods)].copy()
    missing = [method for method in methods if method not in subset["Method"].tolist()]

    if missing:
        raise ValueError(f"Missing methods for '{metric}' comparison: {missing}")

    subset["Method"] = pd.Categorical(
        subset["Method"], categories=methods, ordered=True
    )
    subset = subset.sort_values("Method")

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(subset["Method"].astype(str), subset[metric])

    ax.set_title(title or f"Overall {metric} Comparison", fontsize=15)
    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for bar, value in zip(bars, subset[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_angle_plot(wide_df, methods, metric, output_path, title):
    fig, ax = plt.subplots(figsize=(12, 7))

    for method in methods:
        if method not in wide_df.columns:
            raise ValueError(f"'{method}' is missing from the per-angle table.")

        ax.plot(
            wide_df["Angle"],
            wide_df[method],
            marker="o",
            linewidth=2,
            label=method,
        )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Viewing Angle (degrees)", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xticks(ANGLES)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_dashboard(overall_data, method_a, method_b, output_path):
    """Create a 2x2 dashboard for IoU, Dice, Precision, and Recall."""
    selected = overall_data.set_index("Method")

    missing = [name for name in [method_a, method_b] if name not in selected.index]
    if missing:
        raise ValueError(f"Pairwise comparison is missing methods: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        values = [
            float(selected.loc[method_a, metric]),
            float(selected.loc[method_b, metric]),
        ]
        bars = ax.bar(
            [method_a, method_b],
            values,
        )
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=15)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(f"{method_a} vs {method_b}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PAIRWISE ABLATION COMPARISONS
# ============================================================

PAIRWISE_PAIRS = [
    ("experiment_1_vs_2", "Full Fine-Tuning", "CBAM + BCE + Dice"),
    ("experiment_1_vs_3", "Full Fine-Tuning", "Focal + Dice"),
    ("experiment_2_vs_4", "CBAM + BCE + Dice", "CBAM + Focal + Dice"),
    ("experiment_3_vs_4", "Focal + Dice", "CBAM + Focal + Dice"),
    ("experiment_1_vs_4", "Full Fine-Tuning", "CBAM + Focal + Dice"),
]

print("\nGENERATING PAIRWISE COMPARISONS")

for folder_name, method_a, method_b in PAIRWISE_PAIRS:
    output_path = PAIRWISE_DIR / folder_name / "pairwise_metrics_2x2.png"

    save_pairwise_dashboard(
        overall_df,
        method_a,
        method_b,
        output_path,
    )

    print(f"Saved: {output_path}")


# ============================================================
# OVERALL GROUP COMPARISONS
# ============================================================

print("\nGENERATING OVERALL GROUP COMPARISONS")

for group_name, methods in GROUPS.items():
    group_dir = OVERALL_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    group_df = overall_df[overall_df["Method"].isin(methods)].copy()
    group_df.to_csv(group_dir / f"{group_name}_metrics.csv", index=False)

    for metric in METRICS:
        save_bar_comparison(
            overall_df,
            methods,
            metric,
            group_dir / f"overall_{metric.lower()}_comparison.png",
            title=f"{group_name.replace('_', ' ').title()}: {metric} Comparison",
        )

    # Per-angle plots for the complete set of nine experiments.
    if group_name == "all_experiments":
        save_per_angle_plot(
            per_angle_iou_df,
            methods,
            "IoU",
            group_dir / "iou_vs_viewing_angle_all_nine_experiments.png",
            "IoU vs Viewing Angle - All Nine Experiments",
        )
        save_per_angle_plot(
            per_angle_dice_df,
            methods,
            "Dice",
            group_dir / "dice_vs_viewing_angle_all_nine_experiments.png",
            "Dice vs Viewing Angle - All Nine Experiments",
        )

        # Six-method plot matching the user's reference graph:
        # three fine-tuning strategies and three augmentation strategies.
        six_core_methods = [
            "Full Fine-Tuning",
            "Linear Probing",
            "Partial Fine-Tuning",
            "Geometric Augmentation",
            "Perspective Augmentation",
            "Shadow Augmentation",
        ]

        save_per_angle_plot(
            per_angle_dice_df,
            six_core_methods,
            "Dice",
            group_dir / "dice_vs_viewing_angle.png",
            "Dice vs Viewing Angle",
        )
        save_per_angle_plot(
            per_angle_iou_df,
            six_core_methods,
            "IoU",
            group_dir / "iou_vs_viewing_angle.png",
            "IoU vs Viewing Angle",
        )

    print(f"Saved group: {group_dir}")


# ============================================================
# ALL EXPERIMENTS CONSOLIDATED TABLE AND PLOTS
# ============================================================

all_methods = list(results.keys())
all_dir = OVERALL_DIR / "all_experiments"

for metric in METRICS:
    save_bar_comparison(
        overall_df,
        all_methods,
        metric,
        all_dir / f"all_experiments_{metric.lower()}_comparison.png",
        title=f"All SegFormer Experiments: {metric} Comparison",
    )

# Also save F1 for reference, although the requested comparison graphs
# focus on IoU, Dice, Precision, and Recall.
save_bar_comparison(
    overall_df,
    all_methods,
    "F1",
    all_dir / "all_experiments_f1_comparison.png",
    title="All SegFormer Experiments: F1 Comparison",
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("SEGFORMER COMPARISON GENERATION COMPLETED")
print("=" * 78)

display_columns = [
    column for column in [
        "Method", "IoU", "Dice", "Precision", "Recall", "F1"
    ]
    if column in overall_df.columns
]

print("\nOVERALL RESULTS")
print(overall_df[display_columns].to_string(index=False))

print("\nPER-ANGLE IOU")
print(per_angle_iou_df.to_string(index=False))

print("\nPER-ANGLE DICE")
print(per_angle_dice_df.to_string(index=False))

print("\nAll generated files are saved under:")
print(OUTPUT_DIR)
print("=" * 78)
