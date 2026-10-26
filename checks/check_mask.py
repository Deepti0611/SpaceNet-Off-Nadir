import rasterio
import numpy as np
import matplotlib.pyplot as plt


image_path = "Pan-Sharpen_Atlanta_nadir10_catid_1030010003993E00_743501_3721539.tif"
mask_path = "mask_743501_3721539.png"


# -----------------------------
# Read satellite image
# -----------------------------

with rasterio.open(image_path) as src:
    image = src.read()

print("Image shape:", image.shape)


# -----------------------------
# Convert image for display
# -----------------------------

rgb = image[:3].astype(np.float32)

for i in range(3):

    low = np.percentile(rgb[i], 2)
    high = np.percentile(rgb[i], 98)

    rgb[i] = np.clip(
        (rgb[i] - low) / (high - low),
        0,
        1
    )

rgb = rgb.transpose(1, 2, 0)


# -----------------------------
# Read mask
# -----------------------------

from PIL import Image

mask = np.array(Image.open(mask_path))

# Convert 0-255 mask into 0/1
mask = mask > 0


# -----------------------------
# Display
# -----------------------------

plt.figure(figsize=(10, 10))

plt.imshow(rgb)

# Put the building mask on top
plt.imshow(
    mask,
    alpha=0.4
)

plt.title("Satellite Image + Ground Truth Building Mask")

plt.axis("off")

plt.show()