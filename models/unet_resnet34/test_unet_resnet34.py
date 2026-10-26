import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from pytorch_dataset import SpaceNetDataset
from unet_resnet34_model import get_model


# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Load validation dataset using the existing dataset interface
dataset = SpaceNetDataset(
    metadata_path="dataset/metadata.csv",
    split="validation",
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

# Load pretrained U-Net
model = get_model().to(device)
model.eval()

# Get one sample
images, masks = next(iter(loader))

print("Original images:", images.shape)
print("Original masks:", masks.shape)

# Use first 3 channels for the ImageNet-pretrained encoder
images = images[:, :3, :, :]

# Resize image and mask to 512x512 for the model test
images = F.interpolate(
    images,
    size=(512, 512),
    mode="bilinear",
    align_corners=False,
)

masks = F.interpolate(
    masks,
    size=(512, 512),
    mode="nearest",
)

print("Model input:", images.shape)
print("Resized masks:", masks.shape)

images = images.to(device)

# Forward pass
with torch.no_grad():
    outputs = model(images)

print("Model output:", outputs.shape)