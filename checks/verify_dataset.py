import os
import pandas as pd
import rasterio
from PIL import Image
import numpy as np

DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

metadata = pd.read_csv(METADATA_PATH)

print("=" * 70)
print("FINAL DATASET INTEGRITY CHECK")
print("=" * 70)

print("\nMetadata rows:", len(metadata))

# --------------------------------------------------
# 1. Check expected number of rows
# --------------------------------------------------

expected_rows = 20 * 6

if len(metadata) == expected_rows:
    print(f"✓ Metadata has {expected_rows} rows")
else:
    print(
        f"✗ Expected {expected_rows} rows, "
        f"found {len(metadata)}"
    )

# --------------------------------------------------
# 2. Check splits
# --------------------------------------------------

print("\nSPLIT COUNTS")
print("-" * 40)

print(
    metadata.groupby("split")
    .size()
)

# --------------------------------------------------
# 3. Check angles
# --------------------------------------------------

print("\nANGLE COUNTS")
print("-" * 40)

print(
    metadata.groupby("angle")
    .size()
)

# --------------------------------------------------
# 4. Check each tile has six angles
# --------------------------------------------------

print("\nTILE ANGLE CHECK")
print("-" * 40)

tile_angle_counts = (
    metadata
    .groupby("tile")["angle"]
    .nunique()
)

bad_tiles = tile_angle_counts[
    tile_angle_counts != 6
]

if len(bad_tiles) == 0:
    print("✓ Every tile has all 6 angles")
else:
    print("✗ Problematic tiles:")
    print(bad_tiles)

# --------------------------------------------------
# 5. Check image and mask files
# --------------------------------------------------

missing_images = []
missing_masks = []

for _, row in metadata.iterrows():

    image_path = os.path.join(
        DATASET_DIR,
        row["image_path"]
    )

    mask_path = os.path.join(
        DATASET_DIR,
        row["mask_path"]
    )

    if not os.path.exists(image_path):
        missing_images.append(image_path)

    if not os.path.exists(mask_path):
        missing_masks.append(mask_path)

print("\nFILE EXISTENCE")
print("-" * 40)

if not missing_images:
    print("✓ All image files exist")
else:
    print("✗ Missing images:")
    for path in missing_images:
        print(path)

if not missing_masks:
    print("✓ All mask files exist")
else:
    print("✗ Missing masks:")
    for path in missing_masks:
        print(path)

# --------------------------------------------------
# 6. Check image properties
# --------------------------------------------------

print("\nIMAGE PROPERTY CHECK")
print("-" * 40)

bad_images = []

for _, row in metadata.iterrows():

    image_path = os.path.join(
        DATASET_DIR,
        row["image_path"]
    )

    try:
        with rasterio.open(image_path) as src:

            if (
                src.width != 900
                or src.height != 900
                or src.count != 4
            ):
                bad_images.append(
                    (
                        row["image_id"],
                        src.width,
                        src.height,
                        src.count
                    )
                )

    except Exception as e:

        bad_images.append(
            (
                row["image_id"],
                "ERROR",
                str(e)
            )
        )

if not bad_images:
    print("✓ All images are 900 × 900 × 4")
else:
    print("✗ Problematic images:")
    for item in bad_images:
        print(item)

# --------------------------------------------------
# 7. Check masks
# --------------------------------------------------

print("\nMASK CHECK")
print("-" * 40)

bad_masks = []

for _, row in metadata.iterrows():

    mask_path = os.path.join(
        DATASET_DIR,
        row["mask_path"]
    )

    try:
        mask = np.array(
            Image.open(mask_path)
        )

        unique_values = np.unique(mask)

        if not set(unique_values).issubset({0, 255}):
            bad_masks.append(
                (
                    row["image_id"],
                    unique_values
                )
            )

        if mask.shape != (900, 900):
            bad_masks.append(
                (
                    row["image_id"],
                    f"shape={mask.shape}"
                )
            )

    except Exception as e:

        bad_masks.append(
            (
                row["image_id"],
                "ERROR",
                str(e)
            )
        )

if not bad_masks:
    print("✓ All masks are valid 900 × 900 binary masks")
else:
    print("✗ Problematic masks:")
    for item in bad_masks:
        print(item)

# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n")
print("=" * 70)
print("INTEGRITY CHECK COMPLETE")
print("=" * 70)