import os
import subprocess

import pandas as pd
import numpy as np
from PIL import Image
from shapely import wkt
import rasterio
from rasterio.features import rasterize


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = os.getcwd()

SUMMARY_DIR = os.path.join(PROJECT_DIR, "summaryData")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")

SPLIT_FILE = os.path.join(
    PROJECT_DIR,
    "final_dataset_split.csv"
)

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)


# ============================================================
# SPACE NET COLLECTIONS
# ============================================================

collections = [
    ("Atlanta_nadir7_catid_1030010003D22F00", 7),
    ("Atlanta_nadir10_catid_1030010003993E00", 10),
    ("Atlanta_nadir29_catid_1030010003315300", 29),
    ("Atlanta_nadir39_catid_1030010003832800", 39),
    ("Atlanta_nadir47_catid_10300100033C5200", 47),
    ("Atlanta_nadir53_catid_1030010003193D00", 53),
]


# ============================================================
# TEST MODE
#
# Start with ONE tile.
# After verification, change this to None.
# ============================================================

LIMIT_TILES = None


# ============================================================
# LOAD FINAL SPLIT
# ============================================================

split_df = pd.read_csv(SPLIT_FILE)

all_tiles = split_df["tile"].astype(str).tolist()

if LIMIT_TILES is not None:
    tiles_to_process = all_tiles[:LIMIT_TILES]
else:
    tiles_to_process = all_tiles

print("=" * 70)
print("DATASET BUILD")
print("=" * 70)

print(f"Total tiles in final split: {len(all_tiles)}")
print(f"Tiles to process now: {len(tiles_to_process)}")

print("\nTiles:")
for tile in tiles_to_process:
    split = split_df.loc[
        split_df["tile"] == tile,
        "split"
    ].iloc[0]

    print(f"  {tile} -> {split}")


# ============================================================
# METADATA
# ============================================================

metadata_path = os.path.join(
    DATASET_DIR,
    "metadata.csv"
)

if os.path.exists(metadata_path):
    metadata = pd.read_csv(metadata_path)
else:
    metadata = pd.DataFrame(columns=[
        "tile",
        "angle",
        "category",
        "split",
        "image_id",
        "image_path",
        "mask_path",
        "building_count",
        "image_width",
        "image_height",
        "num_bands"
    ])


# ============================================================
# PROCESS
# ============================================================

successful = 0
failed = 0

for tile_index, tile in enumerate(tiles_to_process, start=1):

    split = split_df.loc[
        split_df["tile"] == tile,
        "split"
    ].iloc[0]

    print("\n")
    print("=" * 70)
    print(f"TILE {tile_index}/{len(tiles_to_process)}: {tile}")
    print(f"Split: {split}")
    print("=" * 70)

    for collection, angle in collections:

        print(f"\n--- {angle}° ---")

        # ----------------------------------------------------
        # Category
        # ----------------------------------------------------

        if angle <= 25:
            category = "nadir"
        elif angle <= 40:
            category = "off-nadir"
        else:
            category = "very-off-nadir"

        # ----------------------------------------------------
        # Image information
        # ----------------------------------------------------

        image_id = f"{collection}_{tile}"

        image_filename = (
            f"Pan-Sharpen_{image_id}.tif"
        )

        mask_filename = (
            f"mask_{angle}deg_{tile}.png"
        )

        image_path = os.path.join(
            IMAGES_DIR,
            image_filename
        )

        mask_path = os.path.join(
            MASKS_DIR,
            mask_filename
        )

        # ----------------------------------------------------
        # Download image
        # ----------------------------------------------------

        s3_path = (
            "s3://spacenet-dataset/spacenet/"
            "SN4_buildings/sample/"
            f"{collection}/Pan-Sharpen/"
            f"{image_filename}"
        )

        if os.path.exists(image_path):

            print("Image already exists. Skipping download.")

        else:

            print("Downloading image...")

            result = subprocess.run([
                "aws",
                "s3",
                "cp",
                "--no-sign-request",
                s3_path,
                image_path
            ])

            if result.returncode != 0:

                print("ERROR: Download failed.")
                failed += 1
                continue

        # ----------------------------------------------------
        # Read image dimensions
        # ----------------------------------------------------

        try:

            with rasterio.open(image_path) as src:

                width = src.width
                height = src.height
                num_bands = src.count

        except Exception as e:

            print("ERROR reading image:", e)
            failed += 1
            continue

        print(
            f"Image: {width} x {height}, "
            f"{num_bands} bands"
        )

        # ----------------------------------------------------
        # Annotation CSV
        # ----------------------------------------------------

        csv_path = os.path.join(
            SUMMARY_DIR,
            f"{collection}_Train.csv"
        )

        df = pd.read_csv(csv_path)

        # Handle occasional Pan-Sharpen_ prefix
        df["clean_image_id"] = (
            df["ImageId"]
            .astype(str)
            .str.replace(
                "Pan-Sharpen_",
                "",
                regex=False
            )
        )

        image_annotations = df[
            df["clean_image_id"] == image_id
        ]

        if len(image_annotations) == 0:

            print(
                "ERROR: No annotations found."
            )

            failed += 1
            continue

        building_count = len(image_annotations)

        print(
            f"Buildings: {building_count}"
        )

        # ----------------------------------------------------
        # Create polygons
        # ----------------------------------------------------

        polygons = []

        for polygon_wkt in image_annotations[
            "PolygonWKT_Pix"
        ]:

            polygon = wkt.loads(polygon_wkt)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            polygons.append(polygon)

        # ----------------------------------------------------
        # Rasterize
        # ----------------------------------------------------

        mask = rasterize(
            [(polygon, 1) for polygon in polygons],
            out_shape=(height, width),
            fill=0,
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        Image.fromarray(
            mask * 255
        ).save(mask_path)

        print(
            f"Mask saved: {mask_filename}"
        )

        print(
            f"Building pixels: "
            f"{np.sum(mask == 1):,}"
        )

        # ----------------------------------------------------
        # Update metadata
        # ----------------------------------------------------

        new_row = {
            "tile": tile,
            "angle": angle,
            "category": category,
            "split": split,
            "image_id": image_id,
            "image_path": os.path.relpath(
                image_path,
                DATASET_DIR
            ).replace("\\", "/"),
            "mask_path": os.path.relpath(
                mask_path,
                DATASET_DIR
            ).replace("\\", "/"),
            "building_count": building_count,
            "image_width": width,
            "image_height": height,
            "num_bands": num_bands
        }

        # Remove an existing identical entry
        metadata = metadata[
            ~(
                (metadata["tile"].astype(str) == tile)
                &
                (
                    metadata["angle"].astype(float)
                    == float(angle)
                )
            )
        ]

        metadata = pd.concat(
            [
                metadata,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

        metadata.to_csv(
            metadata_path,
            index=False
        )

        successful += 1

        print("Completed successfully.")


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BUILD COMPLETE")
print("=" * 70)

print(f"Successful image/mask pairs: {successful}")
print(f"Failed pairs: {failed}")

print(
    f"\nMetadata saved to:\n{metadata_path}"
)

print("\nCurrent metadata:")
print(metadata.to_string(index=False))