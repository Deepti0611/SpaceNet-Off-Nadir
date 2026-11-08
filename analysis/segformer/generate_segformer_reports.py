
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT_DIR / "results" / "segformer"

ANGLES = ["7°", "10°", "29°", "39°", "47°", "53°"]

# Trainable parameter counts verified earlier
TRAINABLE_PARAMETERS = {
    "Full Fine-Tuning": 3714401,
    "Linear Probing": 395009,
    "Partial Fine-Tuning": 3295009
}


# ============================================================
# INPUT FILE DISCOVERY
# ============================================================

def find_file(filename):

    # First check the project root
    direct_path = ROOT_DIR / filename

    if direct_path.exists():
        return direct_path

    # Otherwise search inside the project
    matches = list(ROOT_DIR.rglob(filename))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename}\n"
            f"Place it in the project or update the script."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple copies of {filename} found:\n"
            + "\n".join(str(p) for p in matches)
            + "\nPlease keep one copy or specify its path."
        )

    return matches[0]


# ============================================================
# LOAD JSON FILES
# ============================================================

CONFIGURATIONS = {

    "Full Fine-Tuning": {
        "history": "segformer_history.json",
        "test": "segformer_test_results.json",
        "folder": "full_fine_tuning"
    },

    "Linear Probing": {
        "history": "segformer_linear_probe_history.json",
        "test": "segformer_linear_probe_test_results.json",
        "folder": "linear_probing"
    },

    "Partial Fine-Tuning": {
        "history": "segformer_partial_history.json",
        "test": "segformer_partial_test_results.json",
        "folder": "partial_fine_tuning"
    }
}


def load_results():

    results = {}

    for name, config in CONFIGURATIONS.items():

        history_path = find_file(config["history"])
        test_path = find_file(config["test"])

        with open(history_path, "r") as file:
            history = json.load(file)

        with open(test_path, "r") as file:
            test = json.load(file)

        results[name] = {
            "history": history,
            "test": test,
            "folder": config["folder"]
        }

        print(f"Loaded: {name}")

    return results


# ============================================================
# INDIVIDUAL TRAINING CURVES
# ============================================================

def generate_training_curves(results):

    for name, data in results.items():

        history = data["history"]

        folder = OUTPUT_DIR / data["folder"]
        folder.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(history["train_loss"]) + 1)

        filename_prefix = (
            "segformer_" + data["folder"]
        )

        # ----------------------------------------------------
        # VALIDATION DICE CURVE
        # ----------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            history["validation_dice"],
            marker="o",
            linewidth=2
        )

        plt.title(f"SegFormer-B0 - {name} - Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Dice")
        plt.xticks(list(epochs))
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            folder / f"{filename_prefix}_dice_curve.png",
            dpi=300
        )

        plt.close()

        # ----------------------------------------------------
        # VALIDATION IOU CURVE
        # ----------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            history["validation_iou"],
            marker="o",
            linewidth=2
        )

        plt.title(f"SegFormer-B0 - {name} - Validation IoU")
        plt.xlabel("Epoch")
        plt.ylabel("Validation IoU")
        plt.xticks(list(epochs))
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            folder / f"{filename_prefix}_iou_curve.png",
            dpi=300
        )

        plt.close()

        # ----------------------------------------------------
        # TRAINING AND VALIDATION LOSS CURVE
        # ----------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            history["train_loss"],
            marker="o",
            label="Training Loss"
        )

        plt.plot(
            epochs,
            history["validation_loss"],
            marker="o",
            label="Validation Loss"
        )

        plt.title(f"SegFormer-B0 - {name} - Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(list(epochs))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            folder / f"{filename_prefix}_loss_curve.png",
            dpi=300
        )

        plt.close()

        print(f"Training curves saved: {name}")


# ============================================================
# OVERALL COMPARISON CSV
# ============================================================

def generate_overall_csv(results):

    rows = []

    for name, data in results.items():

        history = data["history"]
        test = data["test"]

        overall = test["overall"]

        best_epoch = test.get("checkpoint_epoch")

        best_val_iou = test.get(
            "checkpoint_validation_iou"
        )

        best_val_dice = None

        if best_epoch is not None:
            best_val_dice = history["validation_dice"][
                best_epoch - 1
            ]

        rows.append({

            "Model": "SegFormer-B0",
            "Configuration": name,

            "Trainable Parameters":
                TRAINABLE_PARAMETERS[name],

            "Best Epoch": best_epoch,

            "Best Validation IoU": best_val_iou,

            "Best Validation Dice": best_val_dice,

            "Test Samples": test["test_samples"],

            "Test IoU": overall["iou"],

            "Test Dice": overall["dice"],

            "Test F1": overall["f1"],

            "Precision": overall["precision"],

            "Recall": overall["recall"]

        })

    output_path = OUTPUT_DIR / "segformer_overall_comparison.csv"

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Overall comparison CSV saved.")


# ============================================================
# PER VIEWING ANGLE CSV
# ============================================================

def generate_per_angle_csv(results):

    rows = []

    for name, data in results.items():

        per_angle = data["test"]["per_angle"]

        for angle in ANGLES:

            metrics = per_angle.get(angle)

            if metrics is None:
                continue

            rows.append({

                "Model": "SegFormer-B0",

                "Configuration": name,

                "Viewing Angle": angle,

                "Images": metrics.get("images"),

                "IoU": metrics.get("iou"),

                "Dice": metrics.get("dice"),

                "F1": metrics.get("f1"),

                "Precision": metrics.get("precision"),

                "Recall": metrics.get("recall")

            })

    output_path = OUTPUT_DIR / "segformer_per_angle_comparison.csv"

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Per-angle comparison CSV saved.")


# ============================================================
# OVERALL IOU COMPARISON GRAPH
# ============================================================

def generate_overall_iou_graph(results):

    names = list(results.keys())

    values = [
        results[name]["test"]["overall"]["iou"]
        for name in names
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(names, values)

    plt.title("SegFormer-B0 - Overall IoU Comparison")
    plt.xlabel("Fine-Tuning Strategy")
    plt.ylabel("Test IoU")

    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "overall_iou_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# OVERALL DICE COMPARISON GRAPH
# ============================================================

def generate_overall_dice_graph(results):

    names = list(results.keys())

    values = [
        results[name]["test"]["overall"]["dice"]
        for name in names
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(names, values)

    plt.title("SegFormer-B0 - Overall Dice Comparison")
    plt.xlabel("Fine-Tuning Strategy")
    plt.ylabel("Test Dice")

    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "overall_dice_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# VIEWING ANGLE COMPARISON GRAPHS
# ============================================================

def generate_angle_graph(results, metric, title, filename):

    plt.figure(figsize=(11, 6))

    for name, data in results.items():

        per_angle = data["test"]["per_angle"]

        values = []

        for angle in ANGLES:

            value = per_angle.get(angle, {}).get(metric)

            values.append(value)

        plt.plot(
            ANGLES,
            values,
            marker="o",
            linewidth=2,
            label=name
        )

    plt.title(title)
    plt.xlabel("Off-Nadir Viewing Angle (degrees)")
    plt.ylabel(metric.upper())

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SEGFORMER-B0 RESULTS GENERATION")
    print("=" * 65)

    results = load_results()

    # Individual training curves
    generate_training_curves(results)

    # CSV files
    generate_overall_csv(results)
    generate_per_angle_csv(results)

    # Overall comparison graphs
    generate_overall_iou_graph(results)
    generate_overall_dice_graph(results)

    # Viewing angle graphs
    generate_angle_graph(
        results,
        metric="iou",
        title="SegFormer-B0 - IoU vs Off-Nadir Viewing Angle",
        filename="iou_vs_viewing_angle.png"
    )

    generate_angle_graph(
        results,
        metric="dice",
        title="SegFormer-B0 - Dice vs Off-Nadir Viewing Angle",
        filename="dice_vs_viewing_angle.png"
    )

    print("\n" + "=" * 65)
    print("ALL RESULTS GENERATED SUCCESSFULLY")
    print("=" * 65)

    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
