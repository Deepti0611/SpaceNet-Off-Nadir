import torch
from torch.utils.data import Dataset

from ml_pipeline.pytorch_dataset import SpaceNetDataset


class AugmentedSpaceNetDataset(Dataset):

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

        # --------------------------------------------------
        # APPLY AUGMENTATION ONLY TO TRAINING DATA
        # --------------------------------------------------

        if self.split == "train":

            # Random horizontal flip
            if torch.rand(1).item() < 0.5:

                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])

            # Random vertical flip
            if torch.rand(1).item() < 0.5:

                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])

            # Random 90-degree rotation
            rotation = torch.randint(
                low=0,
                high=4,
                size=(1,)
            ).item()

            if rotation > 0:

                image = torch.rot90(
                    image,
                    k=rotation,
                    dims=[1, 2]
                )

                mask = torch.rot90(
                    mask,
                    k=rotation,
                    dims=[1, 2]
                )

        return image, mask