import os
import subprocess
import sys

import pandas as pd
import numpy as np
from PIL import Image
from shapely import wkt
import rasterio
from rasterio.features import rasterize


# --------------------------------------------------
# Usage:
#
# python process_tile.py <collection> <tile_id> <angle>
#
# Example:
# python process_tile.py Atlanta_nadir39_catid_1030010003832800 743501_3721539 39
# --------------------------------------------------


if len(sys.argv) != 4:
    print("Usage:")
    print("python process_tile.py <collection> <tile_id> <angle>")
    sys.exit(1)


collection = sys.argv[1]
tile_id = sys.argv[2]
angle = float(sys.argv[3])


# --------------------------------------------------
# Paths
# --------------------------------------------------

project_dir = os.getcwd()

summary_dir = os.path.join(project_dir, "summaryData")
dataset_dir = os.path.join(project_dir, "dataset")

images_dir = os.path.join(dataset_dir, "images")
masks_dir = os.path.join(dataset_dir, "masks")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)


# --------------------------------------------------
# Determine angle category
# --------------------------------------------------

if angle <= 25:
    category = "nadir"
elif angle <= 40:
    category = "off-nadir"
else:
    category = "very-off-nadir"


# --------------------------------------------------
# File names
# --------------------------------------------------

image_id = f"{collection}_{tile_id}"

image_filename = f"Pan-Sharpen_{image_id}.tif"
mask_filename = f"mask_{int(angle)}deg_{tile_id}.png"

image_path = os.path.join(images_dir, image_filename)
mask_path = os.path.join(masks_dir, mask_filename)

csv_filename = f"{collection}_Train.csv"
csv_path = os.path.join(summary_dir, csv_filename)


# --------------------------------------------------
# Check annotation CSV
# --------------------------------------------------

if not os.path.exists(csv_path):
    print("ERROR: Annotation CSV not found:")
    print(csv_path)
    sys.exit(1)


print("\nCollection:", collection)
print("Tile ID:", tile_id)
print("Angle:", angle)
print("Category:", category)


# --------------------------------------------------
# Download image from AWS
# --------------------------------------------------

s3_path = (
    f"s3://spacenet-dataset/spacenet/SN4_buildings/sample/"
    f"{collection}/Pan-Sharpen/{image_filename}"
)

if not os.path.exists(image_path):

    print("\nDownloading image...")

    result = subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            "--no-sign-request",
            s3_path,
            image_path
        ]
    )

    if result.returncode != 0:
        print("ERROR: Image download failed.")
        sys.exit(1)

else:
    print("\nImage already exists. Skipping download.")


# --------------------------------------------------
# Read image dimensions
# --------------------------------------------------

with rasterio.open(image_path) as src:
    width = src.width
    height = src.height


print("Image size:", width, "x", height)


# --------------------------------------------------
# Load annotations
# --------------------------------------------------

print("\nLoading annotations...")

df = pd.read_csv(csv_path)

image_annotations = df[df["ImageId"] == image_id]

if len(image_annotations) == 0:
    print("ERROR: No annotations found for:")
    print(image_id)
    sys.exit(1)


print("Number of buildings:", len(image_annotations))


# --------------------------------------------------
# Convert WKT polygons
# --------------------------------------------------

polygons = []

for polygon_wkt in image_annotations["PolygonWKT_Pix"]:

    polygon = wkt.loads(polygon_wkt)

    if polygon.is_valid:
        polygons.append(polygon)
    else:
        polygons.append(polygon.buffer(0))


print("Polygons loaded:", len(polygons))


# --------------------------------------------------
# Rasterize polygons
# --------------------------------------------------

mask = rasterize(
    [(polygon, 1) for polygon in polygons],
    out_shape=(height, width),
    fill=0,
    dtype=np.uint8
)


# --------------------------------------------------
# Save mask
# --------------------------------------------------

Image.fromarray(mask * 255).save(mask_path)

print("Mask saved to:", mask_path)

print("Background pixels:", np.sum(mask == 0))
print("Building pixels:", np.sum(mask == 1))


# --------------------------------------------------
# Update metadata.csv
# --------------------------------------------------

metadata_path = os.path.join(dataset_dir, "metadata.csv")

new_row = {
    "image_id": tile_id,
    "angle": angle,
    "category": category,
    "image_path": f"images/{image_filename}",
    "mask_path": f"masks/{mask_filename}"
}


if os.path.exists(metadata_path):

    metadata = pd.read_csv(metadata_path)

    # Avoid adding duplicate entries
    duplicate = (
        (metadata["image_id"].astype(str) == str(tile_id))
        & (metadata["angle"].astype(float) == angle)
    ).any()

    if not duplicate:
        metadata = pd.concat(
            [metadata, pd.DataFrame([new_row])],
            ignore_index=True
        )

else:

    metadata = pd.DataFrame([new_row])


metadata.to_csv(metadata_path, index=False)

print("\nMetadata updated.")

print("\nCurrent dataset:")
print(metadata)

print("\nProcessing complete!")