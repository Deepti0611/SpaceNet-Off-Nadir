import torch
import torch.nn.functional as F

from pytorch_dataset import SpaceNetDataset
from segformer_model import create_segformer


def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)

    dice = (
        (2.0 * intersection + smooth)
        / (probs.sum(dim=1) + targets.sum(dim=1) + smooth)
    )

    return 1.0 - dice.mean()


def calculate_iou(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    predictions = (probs > threshold).float()

    predictions = predictions.contiguous().view(predictions.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


print("=" * 70)
print("SEGFORMER + SPACENET DATASET INTEGRATION TEST")
print("=" * 70)

# ---------------------------------------------------------
# 1. Device
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

metadata_path = "dataset/metadata.csv"

train_dataset = SpaceNetDataset(
    metadata_path=metadata_path,
    split="train"
)

print("Train samples:", len(train_dataset))

# ---------------------------------------------------------
# 3. Load one sample
# ---------------------------------------------------------

image, mask = train_dataset[0]

print("\nOriginal sample:")
print("Image shape:", image.shape)
print("Mask shape :", mask.shape)
print("Image dtype:", image.dtype)
print("Mask dtype :", mask.dtype)

# ---------------------------------------------------------
# 4. Add batch dimension
# ---------------------------------------------------------

image = image.unsqueeze(0)
mask = mask.unsqueeze(0)

print("\nAfter adding batch dimension:")
print("Image shape:", image.shape)
print("Mask shape :", mask.shape)

# ---------------------------------------------------------
# 5. Resize to Model 1 resolution
# ---------------------------------------------------------

image = F.interpolate(
    image,
    size=(512, 512),
    mode="bilinear",
    align_corners=False
)

mask = F.interpolate(
    mask,
    size=(512, 512),
    mode="nearest"
)

print("\nAfter resizing:")
print("Image shape:", image.shape)
print("Mask shape :", mask.shape)

# ---------------------------------------------------------
# 6. Load SegFormer
# ---------------------------------------------------------

print("\nLoading SegFormer...")

model = create_segformer()
model = model.to(device)
model.eval()

image = image.to(device)
mask = mask.to(device)

# ---------------------------------------------------------
# 7. Forward pass
# ---------------------------------------------------------

print("\nRunning forward pass...")

with torch.no_grad():
    outputs = model(pixel_values=image)

    logits = outputs.logits

print("Raw SegFormer logits:", logits.shape)

# ---------------------------------------------------------
# 8. Upsample logits to mask size
# ---------------------------------------------------------

logits = F.interpolate(
    logits,
    size=(512, 512),
    mode="bilinear",
    align_corners=False
)

print("Upsampled logits:", logits.shape)

# ---------------------------------------------------------
# 9. Calculate BCE + Dice loss
# ---------------------------------------------------------

bce = torch.nn.BCEWithLogitsLoss()

bce_value = bce(logits, mask)
dice_value = dice_loss(logits, mask)

total_loss = bce_value + dice_value

print("\nLoss:")
print("BCE loss :", bce_value.item())
print("Dice loss:", dice_value.item())
print("Total    :", total_loss.item())

# ---------------------------------------------------------
# 10. Calculate IoU
# ---------------------------------------------------------

iou = calculate_iou(logits, mask)

print("\nIoU:", iou)

# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETED")
print("=" * 70)