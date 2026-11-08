
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================
# 1. LOAD CSV FILES
# =====================================

results_dir = Path("results/segformer/advanced_comparisons")

overall_df = pd.read_csv(
    results_dir / "all_experiments_overall_metrics.csv"
)

dice_df = pd.read_csv(
    results_dir / "all_experiments_per_angle_dice.csv"
)

iou_df = pd.read_csv(
    results_dir / "all_experiments_per_angle_iou.csv"
)

output_dir = results_dir / "line_graphs"
output_dir.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. ALL NINE EXPERIMENTS
# =====================================

methods = [
    "Full Fine-Tuning",
    "Linear Probing",
    "Partial Fine-Tuning",
    "Geometric Augmentation",
    "Perspective Augmentation",
    "Shadow Augmentation",
    "CBAM + BCE + Dice",
    "Focal + Dice",
    "CBAM + Focal + Dice"
]


# =====================================
# 3. OVERALL METRIC LINE GRAPHS
# =====================================

metrics = ["IoU", "Dice", "Precision", "Recall", "F1"]

for metric in metrics:

    plt.figure(figsize=(13, 7))

    plt.plot(
        overall_df["Method"],
        overall_df[metric],
        marker="o",
        linewidth=2,
        markersize=7,
        label=metric
    )

    plt.title(f"Overall {metric} Comparison", fontsize=16)
    plt.xlabel("Experiment", fontsize=12)
    plt.ylabel(metric, fontsize=12)

    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / f"overall_{metric.lower()}.png",
        dpi=300
    )

    plt.close()


# =====================================
# 4. DICE VS VIEWING ANGLE
# =====================================

plt.figure(figsize=(14, 7))

for method in methods:

    plt.plot(
        dice_df["Angle"],
        dice_df[method],
        marker="o",
        linewidth=2,
        markersize=7,
        label=method
    )

plt.title("Dice vs Viewing Angle", fontsize=16)
plt.xlabel("Viewing Angle (degrees)", fontsize=12)
plt.ylabel("Dice", fontsize=12)

plt.xticks([7, 10, 29, 39, 47, 53])
plt.ylim(0, 0.75)

plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=9)

plt.tight_layout()

plt.savefig(
    output_dir / "dice_vs_viewing_angle.png",
    dpi=300
)

plt.close()


# =====================================
# 5. IOU VS VIEWING ANGLE
# =====================================

plt.figure(figsize=(14, 7))

for method in methods:

    plt.plot(
        iou_df["Angle"],
        iou_df[method],
        marker="o",
        linewidth=2,
        markersize=7,
        label=method
    )

plt.title("IoU vs Viewing Angle", fontsize=16)
plt.xlabel("Viewing Angle (degrees)", fontsize=12)
plt.ylabel("IoU", fontsize=12)

plt.xticks([7, 10, 29, 39, 47, 53])
plt.ylim(0, 0.75)

plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=9)

plt.tight_layout()

plt.savefig(
    output_dir / "iou_vs_viewing_angle.png",
    dpi=300
)

plt.close()


# =====================================
# 6. COMPLETION
# =====================================

print("All 7 line graphs generated successfully!")
print("Output folder:", output_dir)
