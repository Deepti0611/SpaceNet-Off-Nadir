import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = (
    "Pan-Sharpen_Atlanta_nadir29_catid_1030010003315300_743501_3721539.tif"
)

mask_path = "mask_29deg_743501_3721539.png"

# Load satellite image
with rasterio.open(image_path) as src:
    image = src.read()

# First 3 bands for RGB display
rgb = image[:3].astype(np.float32)

# Contrast stretching
for i in range(3):
    low = np.percentile(rgb[i], 2)
    high = np.percentile(rgb[i], 98)

    rgb[i] = np.clip(
        (rgb[i] - low) / (high - low),
        0,
        1
    )

rgb = rgb.transpose(1, 2, 0)

# Load mask
mask = np.array(Image.open(mask_path))
mask = mask > 0

# Display overlay
plt.figure(figsize=(10, 10))

plt.imshow(rgb)
plt.imshow(mask, alpha=0.4)

plt.title("29° Satellite Image + Ground Truth Building Mask")
plt.axis("off")

plt.show()