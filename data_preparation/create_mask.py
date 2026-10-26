import pandas as pd
import numpy as np
from PIL import Image
from shapely import wkt
from shapely.geometry import Polygon
from rasterio.features import rasterize
import rasterio


# --------------------------------------------------
# Files
# --------------------------------------------------

csv_path = r"summaryData\Atlanta_nadir10_catid_1030010003993E00_Train.csv"

image_path = r"Pan-Sharpen_Atlanta_nadir10_catid_1030010003993E00_743501_3721539.tif"

mask_path = "mask_743501_3721539.png"


# --------------------------------------------------
# Image information
# --------------------------------------------------

with rasterio.open(image_path) as src:
    width = src.width
    height = src.height

print("Image size:", width, "x", height)


# --------------------------------------------------
# Read annotations
# --------------------------------------------------

df = pd.read_csv(csv_path)

image_id = "Atlanta_nadir10_catid_1030010003993E00_743501_3721539"

# Keep only annotations belonging to our image
image_annotations = df[df["ImageId"] == image_id]

print("Number of buildings:", len(image_annotations))


# --------------------------------------------------
# Convert WKT polygons into Shapely polygons
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
# Rasterize polygons into a binary mask
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