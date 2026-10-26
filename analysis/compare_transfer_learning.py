import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\SpaceNet_Project"

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "transfer_learning_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# RESULT FILES
# ============================================================

result_files = {
    "Linear Probing": os.path.join(
        RESULTS_DIR,
        "deeplabv3plus_resnet50_linear_probe",
        "deeplabv3plus_resnet50_linear_probe_test_results.json"
    ),

    "Partial Fine-Tuning": os.path.join(
        RESULTS_DIR,
        "deeplabv3plus_resnet50_partial",
        "deeplabv3plus_resnet50_partial_test_results.json"
    ),

    "Full Fine-Tuning": os.path.join(
        RESULTS_DIR,
        "deeplabv3plus_resnet50",
        "deeplabv3plus_resnet50_test_results.json"
    )
}


# ============================================================
# TRAINABLE PARAMETER COUNTS
# ============================================================

trainable_parameters = {
    "Linear Probing": 3169553,
    "Partial Fine-Tuning": 25232657,
    "Full Fine-Tuning": 26677585
}


# ============================================================
# LOAD JSON RESULTS
# ============================================================

results = {}

for method, filepath in result_files.items():

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Result file not found:\n{filepath}"
        )

    with open(filepath, "r") as f:
        results[method] = json.load(f)


# ============================================================
# OVERALL COMPARISON TABLE
# ============================================================

overall_rows = []

for method, data in results.items():

    metrics = data["overall"]

    overall_rows.append({
        "Method": method,
        "Trainable Parameters": trainable_parameters[method],
        "IoU": metrics["iou"],
        "Dice": metrics["dice"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"]
    })


overall_df = pd.DataFrame(overall_rows)


# Sort by IoU
overall_df = overall_df.sort_values(
    by="IoU",
    ascending=False
).reset_index(drop=True)


# Save CSV
overall_csv = os.path.join(
    OUTPUT_DIR,
    "transfer_learning_overall_comparison.csv"
)

overall_df.to_csv(
    overall_csv,
    index=False
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print()
print("=" * 80)
print("TRANSFER LEARNING - OVERALL TEST COMPARISON")
print("=" * 80)

print(
    overall_df.to_string(
        index=False,
        formatters={
            "IoU": "{:.4f}".format,
            "Dice": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)


# ============================================================
# PLOT 1: OVERALL IoU
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    overall_df["Method"],
    overall_df["IoU"]
)

plt.xlabel("Transfer Learning Strategy")
plt.ylabel("IoU")
plt.title("Test IoU Comparison of Transfer Learning Strategies")

plt.ylim(
    0,
    max(overall_df["IoU"]) * 1.2
)

plt.xticks(rotation=15)

plt.tight_layout()

iou_plot = os.path.join(
    OUTPUT_DIR,
    "overall_iou_comparison.png"
)

plt.savefig(
    iou_plot,
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2: OVERALL DICE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    overall_df["Method"],
    overall_df["Dice"]
)

plt.xlabel("Transfer Learning Strategy")
plt.ylabel("Dice")
plt.title("Test Dice Comparison of Transfer Learning Strategies")

plt.ylim(
    0,
    max(overall_df["Dice"]) * 1.2
)

plt.xticks(rotation=15)

plt.tight_layout()

dice_plot = os.path.join(
    OUTPUT_DIR,
    "overall_dice_comparison.png"
)

plt.savefig(
    dice_plot,
    dpi=300
)

plt.close()


# ============================================================
# PER-ANGLE DATA
# ============================================================

angles = sorted(
    {
        float(angle)
        for data in results.values()
        for angle in data["per_angle"].keys()
    }
)


angle_rows = []

for angle in angles:

    for method, data in results.items():

        angle_key = str(int(angle))

        metrics = data["per_angle"][angle_key]

        angle_rows.append({
            "Angle": angle,
            "Method": method,
            "IoU": metrics["iou"],
            "Dice": metrics["dice"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"]
        })


angle_df = pd.DataFrame(angle_rows)


# Save angle CSV
angle_csv = os.path.join(
    OUTPUT_DIR,
    "transfer_learning_per_angle_comparison.csv"
)

angle_df.to_csv(
    angle_csv,
    index=False
)


# ============================================================
# PRINT PER-ANGLE IoU
# ============================================================

print()
print("=" * 80)
print("IoU BY VIEWING ANGLE")
print("=" * 80)

iou_pivot = angle_df.pivot(
    index="Angle",
    columns="Method",
    values="IoU"
)

print(
    iou_pivot.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# PLOT 3: IoU VS VIEWING ANGLE
# ============================================================

plt.figure(figsize=(9, 5))

for method in result_files.keys():

    method_data = angle_df[
        angle_df["Method"] == method
    ].sort_values("Angle")

    plt.plot(
        method_data["Angle"],
        method_data["IoU"],
        marker="o",
        label=method
    )

plt.xlabel("Off-Nadir Viewing Angle (degrees)")
plt.ylabel("IoU")
plt.title("IoU vs Off-Nadir Viewing Angle")

plt.xticks(angles)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

angle_iou_plot = os.path.join(
    OUTPUT_DIR,
    "iou_vs_viewing_angle.png"
)

plt.savefig(
    angle_iou_plot,
    dpi=300
)

plt.close()


# ============================================================
# PLOT 4: DICE VS VIEWING ANGLE
# ============================================================

plt.figure(figsize=(9, 5))

for method in result_files.keys():

    method_data = angle_df[
        angle_df["Method"] == method
    ].sort_values("Angle")

    plt.plot(
        method_data["Angle"],
        method_data["Dice"],
        marker="o",
        label=method
    )

plt.xlabel("Off-Nadir Viewing Angle (degrees)")
plt.ylabel("Dice")
plt.title("Dice vs Off-Nadir Viewing Angle")

plt.xticks(angles)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

angle_dice_plot = os.path.join(
    OUTPUT_DIR,
    "dice_vs_viewing_angle.png"
)

plt.savefig(
    angle_dice_plot,
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

print()
print("Files generated:")

print(overall_csv)
print(angle_csv)
print(iou_plot)
print(dice_plot)
print(angle_iou_plot)
print(angle_dice_plot)