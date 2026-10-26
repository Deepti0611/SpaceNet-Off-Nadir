import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = (
    "dataset/images/"
    "Pan-Sharpen_Atlanta_nadir39_catid_1030010003832800_743501_3721539.tif"
)

mask_path = (
    "dataset/masks/"
    "mask_39deg_743501_3721539.png"
)

with rasterio.open(image_path) as src:
    image = src.read()

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

mask = np.array(Image.open(mask_path))
mask = mask > 0

plt.figure(figsize=(10, 10))

plt.imshow(rgb)
plt.imshow(mask, alpha=0.4)

plt.title("39° Satellite Image + Ground Truth Building Mask")
plt.axis("off")

plt.show()