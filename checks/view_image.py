import rasterio
import numpy as np
import matplotlib.pyplot as plt

image_path = "Pan-Sharpen_Atlanta_nadir10_catid_1030010003993E00_743501_3721539.tif"

with rasterio.open(image_path) as src:

    print("Number of bands:", src.count)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Data types:", src.dtypes)

    image = src.read()

print("Original shape:", image.shape)


# -----------------------------------
# Convert image for visualization
# -----------------------------------

# Take the first 3 channels
rgb = image[:3].astype(np.float32)

# Contrast stretching for each channel
for i in range(3):

    low = np.percentile(rgb[i], 2)
    high = np.percentile(rgb[i], 98)

    rgb[i] = np.clip(
        (rgb[i] - low) / (high - low),
        0,
        1
    )

# Change from:
# (channels, height, width)
#
# to:
# (height, width, channels)

rgb = rgb.transpose(1, 2, 0)


# -----------------------------------
# Display
# -----------------------------------

plt.figure(figsize=(10, 10))

plt.imshow(rgb)

plt.title("SpaceNet 4 - Pan-Sharpened Satellite Image")

plt.axis("off")

plt.show()