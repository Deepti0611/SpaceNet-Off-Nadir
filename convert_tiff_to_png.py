import rasterio
import numpy as np
from PIL import Image
from pathlib import Path
import re

# --------------------------------------------------
# Settings
# --------------------------------------------------

image_folder = Path("dataset/images")
output_folder = Path("dataset/images/converted")

# Geographic location ID
location_id = "733601_3724689"

# The six viewing angles
angles = ["7", "10", "29", "39", "47", "53"]

output_folder.mkdir(exist_ok=True)

# --------------------------------------------------
# Convert one TIFF to RGB PNG
# --------------------------------------------------

def convert_tiff_to_png(input_tif, output_png):

    with rasterio.open(input_tif) as src:

        print(f"Reading: {input_tif.name}")

        # SpaceNet image is:
        # Band 1 = Blue
        # Band 2 = Green
        # Band 3 = Red
        # Band 4 = NIR

        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

        rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)

        # Normalize each band for visualization
        for i in range(3):

            band = rgb[:, :, i]

            low = np.percentile(band, 2)
            high = np.percentile(band, 98)

            if high > low:
                band = (band - low) / (high - low)

            band = np.clip(band, 0, 1)

            rgb[:, :, i] = band * 255

        rgb = rgb.astype(np.uint8)

        Image.fromarray(rgb).save(output_png)

        print(f"Saved: {output_png}")


# --------------------------------------------------
# Find and convert all six angles
# --------------------------------------------------

for angle in angles:

    matches = list(image_folder.glob(
        f"*nadir{angle}*{location_id}.tif"
    ))

    if len(matches) == 0:

        print(f"WARNING: No image found for {angle}°")

        continue

    input_tif = matches[0]

    output_png = output_folder / f"satellite_{angle}deg_{location_id}.png"

    convert_tiff_to_png(input_tif, output_png)


print("\nConversion complete!")
print(f"Images saved in: {output_folder}")