
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT_DIR / "results" / "segformer" / "segformer_comparisions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT FILES
# ============================================================

EXPERIMENTS = {
    "Full Fine-Tuning":
        "segformer_test_results.json",

    "Linear Probing":
        "segformer_linear_probe_test_results.json",

    "Partial Fine-Tuning":
        "segformer_partial_test_results.json",

    "Geometric Augmentation":
        "segformer_geometric_test_results.json",

    "Perspective Augmentation":
        "segformer_perspective_test_results.json",

    "Shadow Augmentation":
        "segformer_shadow_test_results.json"
}


# ============================================================
# FIND AND LOAD JSON FILES
# ============================================================

def find_result_file(filename):

    matches = list(ROOT_DIR.rglob(filename))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename} inside {ROOT_DIR}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files named {filename} found:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


results = {}

for experiment, filename in EXPERIMENTS.items():

    file_path = find_result_file(filename)

    with open(file_path, "r", encoding="utf-8") as file:
        results[experiment] = json.load(file)

    print(f"Loaded: {experiment}")
    print(f"       {file_path}")


# ============================================================
# EXTRACT OVERALL METRICS
# ============================================================

overall_rows = []

for experiment, result in results.items():

    metrics = result["overall"]

    overall_rows.append({
        "Method": experiment,
        "IoU": metrics["iou"],
        "Dice": metrics["dice"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"]
    })


overall_df = pd.DataFrame(overall_rows)

overall_df.to_csv(
    OUTPUT_DIR / "overall_results.csv",
    index=False
)


# ============================================================
# EXTRACT PER-ANGLE METRICS
# ============================================================

angles = ["7°", "10°", "29°", "39°", "47°", "53°"]

iou_data = {"Angle": [7, 10, 29, 39, 47, 53]}
dice_data = {"Angle": [7, 10, 29, 39, 47, 53]}

for experiment, result in results.items():

    angle_results = result["per_angle"]

    iou_values = []
    dice_values = []

    for angle in angles:

        if angle not in angle_results:
            raise ValueError(
                f"{angle} missing from {experiment} results"
            )

        metrics = angle_results[angle]

        if metrics["iou"] is None or metrics["dice"] is None:
            raise ValueError(
                f"Missing metric for {experiment}, {angle}"
            )

        iou_values.append(metrics["iou"])
        dice_values.append(metrics["dice"])

    iou_data[experiment] = iou_values
    dice_data[experiment] = dice_values


iou_df = pd.DataFrame(iou_data)
dice_df = pd.DataFrame(dice_data)

iou_df.to_csv(
    OUTPUT_DIR / "per_angle_iou.csv",
    index=False
)

dice_df.to_csv(
    OUTPUT_DIR / "per_angle_dice.csv",
    index=False
)


# ============================================================
# GRAPH 1: DICE VS VIEWING ANGLE
# ============================================================

plt.figure(figsize=(13, 7))

for experiment in EXPERIMENTS:

    plt.plot(
        dice_df["Angle"],
        dice_df[experiment],
        marker="o",
        linewidth=2,
        label=experiment
    )

plt.title("Dice vs Viewing Angle", fontsize=16)
plt.xlabel("Viewing Angle (degrees)", fontsize=12)
plt.ylabel("Dice", fontsize=12)

plt.xticks([7, 10, 29, 39, 47, 53])
plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=10)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "dice_vs_viewing_angle.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2: IOU VS VIEWING ANGLE
# ============================================================

plt.figure(figsize=(13, 7))

for experiment in EXPERIMENTS:

    plt.plot(
        iou_df["Angle"],
        iou_df[experiment],
        marker="o",
        linewidth=2,
        label=experiment
    )

plt.title("IoU vs Viewing Angle", fontsize=16)
plt.xlabel("Viewing Angle (degrees)", fontsize=12)
plt.ylabel("IoU", fontsize=12)

plt.xticks([7, 10, 29, 39, 47, 53])
plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=10)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "iou_vs_viewing_angle.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3: OVERALL DICE COMPARISON
# ============================================================

plt.figure(figsize=(13, 7))

plt.bar(
    overall_df["Method"],
    overall_df["Dice"]
)

plt.title("Overall Dice Comparison", fontsize=16)
plt.xlabel("Experiment", fontsize=12)
plt.ylabel("Dice", fontsize=12)

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overall_dice_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 4: OVERALL IOU COMPARISON
# ============================================================

plt.figure(figsize=(13, 7))

plt.bar(
    overall_df["Method"],
    overall_df["IoU"]
)

plt.title("Overall IoU Comparison", fontsize=16)
plt.xlabel("Experiment", fontsize=12)
plt.ylabel("IoU", fontsize=12)

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overall_iou_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SEGFORMER EXPERIMENT COMPARISON COMPLETED")
print("=" * 65)

print("\nOVERALL RESULTS")
print(overall_df.to_string(index=False))

print("\nPER-ANGLE IOU")
print(iou_df.to_string(index=False))

print("\nPER-ANGLE DICE")
print(dice_df.to_string(index=False))

print("\nAll outputs saved in:")
print(OUTPUT_DIR)

print("=" * 65)
