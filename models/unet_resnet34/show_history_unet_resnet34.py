import json

with open("unet_resnet34_history.json", "r") as f:
    history = json.load(f)

for i in range(len(history["train_loss"])):
    print(
        f"Epoch {i + 1}: "
        f"Train Loss={history['train_loss'][i]:.4f} | "
        f"Val Loss={history['validation_loss'][i]:.4f} | "
        f"IoU={history['validation_iou'][i]:.4f} | "
        f"Dice={history['validation_dice'][i]:.4f}"
    )