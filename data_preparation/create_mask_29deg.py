import pandas as pd
import numpy as np
from PIL import Image
from shapely import wkt
import rasterio
from rasterio.features import rasterize

csv_path = r"summaryData\Atlanta_nadir29_catid_1030010003315300_Train.csv"

image_path = (
    r"Pan-Sharpen_Atlanta_nadir29_catid_1030010003315300_743501_3721539.tif"
)

mask_path = "mask_29deg_743501_3721539.png"

image_id = (
    "Atlanta_nadir29_catid_1030010003315300_743501_3721539"
)

# Get image dimensions
with rasterio.open(image_path) as src:
    width = src.width
    height = src.height

# Load annotations
df = pd.read_csv(csv_path)

image_annotations = df[df["ImageId"] == image_id]

# Convert WKT polygons
polygons = []

for polygon_wkt in image_annotations["PolygonWKT_Pix"]:

    polygon = wkt.loads(polygon_wkt)

    if polygon.is_valid:
        polygons.append(polygon)
    else:
        polygons.append(polygon.buffer(0))

# Rasterize
mask = rasterize(
    [(polygon, 1) for polygon in polygons],
    out_shape=(height, width),
    fill=0,
    dtype=np.uint8
)

# Save mask
Image.fromarray(mask * 255).save(mask_path)

print("Image size:", width, "x", height)
print("Number of buildings:", len(image_annotations))
print("Polygons loaded:", len(polygons))
print("Background pixels:", np.sum(mask == 0))
print("Building pixels:", np.sum(mask == 1))
print("Mask saved to:", mask_path)