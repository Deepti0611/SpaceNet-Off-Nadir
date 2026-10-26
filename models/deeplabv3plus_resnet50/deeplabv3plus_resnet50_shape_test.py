import torch
import torch.nn.functional as F

from deeplabv3plus_resnet50_model import get_model
from pytorch_dataset import SpaceNetDataset


METADATA_PATH = "dataset/metadata.csv"
IMAGE_SIZE = 512


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# Load validation dataset
dataset = SpaceNetDataset(
    METADATA_PATH,
    split="validation"
)

image, mask = dataset[0]

print("Original image shape:", image.shape)
print("Original mask shape:", mask.shape)


# Add batch dimension
image = image.unsqueeze(0)
mask = mask.unsqueeze(0)

print("Batch image shape:", image.shape)
print("Batch mask shape:", mask.shape)


# Use first 3 channels
image = image[:, :3, :, :]

print("After selecting 3 channels:", image.shape)


# Resize image
image = F.interpolate(
    image,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False
)

# Resize mask
mask = F.interpolate(
    mask,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="nearest"
)

print("Model input shape:", image.shape)
print("Resized mask shape:", mask.shape)


# Load model
model = get_model().to(device)
model.eval()

image = image.to(device)

# Forward pass
with torch.no_grad():
    output = model(image)

print("Model output shape:", output.shape)