import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import torch
import matplotlib.pyplot as plt

from ml_pipeline.pytorch_dataset import SpaceNetDataset
from ml_pipeline.augmented_dataset import AugmentedSpaceNetDataset


METADATA_PATH = "dataset/metadata.csv"

# Load original training dataset
original_dataset = SpaceNetDataset(
    metadata_path=METADATA_PATH,
    split="train"
)

# Load geometrically augmented training dataset
augmented_dataset = AugmentedSpaceNetDataset(
    metadata_path=METADATA_PATH,
    split="train"
)

# Select the same sample
index = 0

original_image, original_mask = original_dataset[index]
augmented_image, augmented_mask = augmented_dataset[index]

# Print tensor details
print("\nORIGINAL SAMPLE")
print("Image shape:", original_image.shape)
print("Mask shape:", original_mask.shape)

print("\nAUGMENTED SAMPLE")
print("Image shape:", augmented_image.shape)
print("Mask shape:", augmented_mask.shape)

print("\nMask values:", torch.unique(augmented_mask))
print("Image range:", augmented_image.min().item(),
      augmented_image.max().item())

# Convert tensors for plotting
original_image = original_image.permute(1, 2, 0).numpy()
augmented_image = augmented_image.permute(1, 2, 0).numpy()

original_mask = original_mask.squeeze().numpy()
augmented_mask = augmented_mask.squeeze().numpy()

# Visualize
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(original_mask, cmap="gray")
plt.title("Original Mask")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(augmented_image)
plt.title("Geometrically Augmented Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(augmented_mask, cmap="gray")
plt.title("Geometrically Augmented Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
