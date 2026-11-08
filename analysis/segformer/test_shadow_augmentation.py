
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from ml_pipeline.pytorch_dataset import SpaceNetDataset
from ml_pipeline.shadow_augmented_dataset import ShadowAugmentedSpaceNetDataset


# ------------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------------

original_dataset = SpaceNetDataset(
    metadata_path="dataset/metadata.csv",
    split="train"
)

augmented_dataset = ShadowAugmentedSpaceNetDataset(
    metadata_path="dataset/metadata.csv",
    split="train"
)


# ------------------------------------------------------------
# SELECT SAMPLE
# ------------------------------------------------------------

index = 0

original_image, original_mask = original_dataset[index]
augmented_image, augmented_mask = augmented_dataset[index]


# ------------------------------------------------------------
# PRINT TENSOR INFORMATION
# ------------------------------------------------------------

print("\nORIGINAL SAMPLE")
print("Image shape:", original_image.shape)
print("Mask shape:", original_mask.shape)

print("\nSHADOW AUGMENTED SAMPLE")
print("Image shape:", augmented_image.shape)
print("Mask shape:", augmented_mask.shape)

print("\nMask values:", torch.unique(augmented_mask))

print(
    "Image range:",
    augmented_image.min().item(),
    augmented_image.max().item()
)


# ------------------------------------------------------------
# CONVERT TENSORS FOR VISUALIZATION
# ------------------------------------------------------------

original_image = original_image.permute(1, 2, 0).numpy()
original_mask = original_mask.squeeze().numpy()

augmented_image = augmented_image.permute(1, 2, 0).numpy()
augmented_mask = augmented_mask.squeeze().numpy()


# ------------------------------------------------------------
# VISUALIZE
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(original_image)
axes[0, 0].set_title("Original Image")

axes[0, 1].imshow(original_mask, cmap="gray")
axes[0, 1].set_title("Original Mask")

axes[1, 0].imshow(augmented_image)
axes[1, 0].set_title("Shadow Augmented Image")

axes[1, 1].imshow(augmented_mask, cmap="gray")
axes[1, 1].set_title("Shadow Augmented Mask")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
