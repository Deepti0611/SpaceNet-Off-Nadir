import json
import matplotlib.pyplot as plt

HISTORY_PATH = "models/deeplabv3plus_resnet50/deeplabv3plus_resnet50_history.json"

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

# -----------------------------
# Loss curve
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DeepLabV3+ ResNet50 - Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "results/deeplabv3plus_resnet50_loss_curve.png",
    dpi=300
)

plt.show()


# -----------------------------
# IoU curve
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, history["val_iou"], marker="o")

plt.xlabel("Epoch")
plt.ylabel("Validation IoU")
plt.title("DeepLabV3+ ResNet50 - Validation IoU")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "results/deeplabv3plus_resnet50_iou_curve.png",
    dpi=300
)

plt.show()


# -----------------------------
# Dice curve
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(epochs, history["val_dice"], marker="o")

plt.xlabel("Epoch")
plt.ylabel("Validation Dice")
plt.title("DeepLabV3+ ResNet50 - Validation Dice")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "results/deeplabv3plus_resnet50_dice_curve.png",
    dpi=300
)

plt.show()