import os
import pandas as pd
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class SpaceNetDataset(Dataset):

    def __init__(self, metadata_path, split):

        self.metadata = pd.read_csv(metadata_path)

        # Keep only the requested split
        self.metadata = self.metadata[
            self.metadata["split"] == split
        ].reset_index(drop=True)

        print(f"{split} samples: {len(self.metadata)}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        # --------------------------------------------------
        # IMAGE
        # --------------------------------------------------

        image_path = os.path.join(
            "dataset",
            row["image_path"]
        )

        with rasterio.open(image_path) as src:
            image = src.read()

        # Original bands:
        # Band 1 = Blue
        # Band 2 = Green
        # Band 3 = Red
        # Band 4 = NIR
        #
        # Convert B,G,R -> R,G,B

        image = image[[2, 1, 0], :, :]

        # Convert uint16 -> float32
        image = image.astype(np.float32)

        # --------------------------------------------------
        # NORMALIZATION
        # --------------------------------------------------

        for band in range(3):

            band_data = image[band]

            min_value = band_data.min()
            max_value = band_data.max()

            if max_value > min_value:

                image[band] = (
                    (band_data - min_value)
                    / (max_value - min_value)
                )

            else:
                image[band] = 0

        # --------------------------------------------------
        # MASK
        # --------------------------------------------------

        mask_path = os.path.join(
            "dataset",
            row["mask_path"]
        )

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # Convert:
        # 0   -> 0
        # 255 -> 1

        mask = (mask > 0).astype(np.float32)

        # --------------------------------------------------
        # NUMPY -> PYTORCH
        # --------------------------------------------------

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Add channel dimension to mask
        mask = mask.unsqueeze(0)

        return image, mask


# ==========================================================
# DATASET + DATALOADER TEST
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # CREATE DATASETS
    # ------------------------------------------------------

    train_dataset = SpaceNetDataset(
        metadata_path="dataset/metadata.csv",
        split="train"
    )

    val_dataset = SpaceNetDataset(
        metadata_path="dataset/metadata.csv",
        split="validation"
    )

    test_dataset = SpaceNetDataset(
        metadata_path="dataset/metadata.csv",
        split="test"
    )

    # ------------------------------------------------------
    # CREATE DATALOADERS
    # ------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False
    )

    # ------------------------------------------------------
    # GET ONE TRAINING BATCH
    # ------------------------------------------------------

    images, masks = next(iter(train_loader))

    # ------------------------------------------------------
    # PRINT INFORMATION
    # ------------------------------------------------------

    print()
    print("DATASET SIZES")
    print("-------------------------")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    print()
    print("ONE TRAINING BATCH")
    print("-------------------------")
    print("Images shape:", images.shape)
    print("Masks shape :", masks.shape)
    print("Images dtype:", images.dtype)
    print("Masks dtype :", masks.dtype)
    print("Mask values :", torch.unique(masks))

    # ------------------------------------------------------
    # VISUAL CHECK
    # ------------------------------------------------------

    image = images[0].permute(1, 2, 0).numpy()
    mask = masks[0].squeeze(0).numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Satellite Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()