import torch
from torch.utils.data import Dataset
from ml_pipeline.pytorch_dataset import SpaceNetDataset


class ShadowAugmentedSpaceNetDataset(Dataset):

    def __init__(self, metadata_path, split):
        self.base_dataset = SpaceNetDataset(
            metadata_path=metadata_path,
            split=split
        )
        self.split = split

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):

        image, mask = self.base_dataset[idx]

        # Apply shadow augmentation ONLY during training
        if self.split == "train":

            # 50% probability of adding a shadow
            if torch.rand(1).item() < 0.5:

                _, height, width = image.shape

                # Random shadow direction
                direction = torch.randint(
                    low=0,
                    high=4,
                    size=(1,)
                ).item()

                # Random shadow strength
                shadow_strength = torch.empty(1).uniform_(
                    0.5, 0.8
                ).item()

                # Create shadow mask
                shadow = torch.ones(
                    (height, width),
                    dtype=image.dtype
                )

                if direction == 0:
                    # Left side shadow
                    shadow[:, :width // 2] = shadow_strength

                elif direction == 1:
                    # Right side shadow
                    shadow[:, width // 2:] = shadow_strength

                elif direction == 2:
                    # Top shadow
                    shadow[:height // 2, :] = shadow_strength

                else:
                    # Bottom shadow
                    shadow[height // 2:, :] = shadow_strength

                # Apply shadow to all RGB channels
                shadow = shadow.unsqueeze(0)

                image = image * shadow

                # Keep image values valid
                image = torch.clamp(image, 0.0, 1.0)

        # IMPORTANT:
        # Mask is NOT modified by shadow augmentation.
        return image, mask