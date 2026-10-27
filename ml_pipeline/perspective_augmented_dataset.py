
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from ml_pipeline.pytorch_dataset import SpaceNetDataset


class PerspectiveAugmentedSpaceNetDataset(Dataset):

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

        # Apply augmentation only to training data
        if self.split == "train":

            # Randomly choose whether to apply perspective distortion
            if torch.rand(1).item() < 0.5:

                _, height, width = image.shape

                # Random perspective parameters
                distortion = 0.15

                # Maximum displacement
                max_dx = int(width * distortion)
                max_dy = int(height * distortion)

                # Random displacement for the four corners
                start_points = torch.tensor([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=torch.float32)

                end_points = start_points.clone()

                end_points[:, 0] += torch.randint(
                    -max_dx,
                    max_dx + 1,
                    (4,)
                ).float()

                end_points[:, 1] += torch.randint(
                    -max_dy,
                    max_dy + 1,
                    (4,)
                ).float()

                # Keep points inside the image
                end_points[:, 0] = end_points[:, 0].clamp(
                    0, width - 1
                )

                end_points[:, 1] = end_points[:, 1].clamp(
                    0, height - 1
                )

                # ------------------------------------------------
                # Perspective transform
                # ------------------------------------------------

                # Convert image and mask to batch format
                image_batch = image.unsqueeze(0)
                mask_batch = mask.unsqueeze(0)

                # Create normalized sampling grid
                theta = torch.zeros(
                    (1, 2, 3),
                    dtype=torch.float32
                )

                theta[:, 0, 0] = 1.0
                theta[:, 1, 1] = 1.0

                # Random affine approximation of viewpoint change
                theta[:, 0, 1] = torch.empty(1).uniform_(
                    -0.15, 0.15
                )

                theta[:, 1, 0] = torch.empty(1).uniform_(
                    -0.10, 0.10
                )

                theta[:, 0, 2] = torch.empty(1).uniform_(
                    -0.10, 0.10
                )

                theta[:, 1, 2] = torch.empty(1).uniform_(
                    -0.10, 0.10
                )

                grid = F.affine_grid(
                    theta,
                    image_batch.size(),
                    align_corners=False
                )

                image_batch = F.grid_sample(
                    image_batch,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False
                )

                mask_batch = F.grid_sample(
                    mask_batch,
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False
                )

                image = image_batch.squeeze(0)
                mask = mask_batch.squeeze(0)

        return image, mask
