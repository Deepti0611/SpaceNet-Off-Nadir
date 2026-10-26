import rasterio
import numpy as np
import matplotlib.pyplot as plt

images = [
    (
        "Pan-Sharpen_Atlanta_nadir7_catid_1030010003D22F00_743501_3721539.tif",
        "7°"
    ),
    (
        "Pan-Sharpen_Atlanta_nadir10_catid_1030010003993E00_743501_3721539.tif",
        "~10°"
    ),
    (
        "Pan-Sharpen_Atlanta_nadir29_catid_1030010003315300_743501_3721539.tif",
        "29°"
    )
]

plt.figure(figsize=(18, 6))

for i, (image_path, title) in enumerate(images):

    with rasterio.open(image_path) as src:
        image = src.read()

    rgb = image[:3].astype(np.float32)

    for band in range(3):
        low = np.percentile(rgb[band], 2)
        high = np.percentile(rgb[band], 98)

        rgb[band] = np.clip(
            (rgb[band] - low) / (high - low),
            0,
            1
        )

    rgb = rgb.transpose(1, 2, 0)

    plt.subplot(1, 3, i + 1)
    plt.imshow(rgb)
    plt.title(f"Viewing angle: {title}")
    plt.axis("off")

plt.tight_layout()
plt.show()