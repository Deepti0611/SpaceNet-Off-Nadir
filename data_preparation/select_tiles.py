import pandas as pd
import re
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------

csv_path = (
    r"summaryData\Atlanta_nadir7_catid_1030010003D22F00_Train.csv"
)

output_path = "selected_tiles.csv"

NUMBER_OF_TILES = 20


# --------------------------------------------------
# Load annotation CSV
# --------------------------------------------------

df = pd.read_csv(csv_path)

image_ids = df["ImageId"].dropna().astype(str).unique()

print("Raw unique ImageIds:", len(image_ids))


# --------------------------------------------------
# Extract tile coordinates
#
# ImageId ends with:
#
#     xxxxxx_yyyyyyy
#
# Example:
#     743051_3735939
# --------------------------------------------------

tiles = []

for image_id in image_ids:

    match = re.search(r"_(\d+)_(\d+)$", image_id)

    if match:

        x = int(match.group(1))
        y = int(match.group(2))

        tile_id = f"{x}_{y}"

        tiles.append({
            "tile_id": tile_id,
            "x": x,
            "y": y,
            "original_image_id": image_id
        })


tiles_df = pd.DataFrame(tiles)

# Remove duplicate tile coordinates
tiles_df = tiles_df.drop_duplicates(
    subset=["tile_id"]
).reset_index(drop=True)


print("Valid tile IDs:", len(tiles_df))


# --------------------------------------------------
# Spatially spread selection
#
# Divide the available area into a grid and select
# one tile from each grid cell.
# --------------------------------------------------

x_min = tiles_df["x"].min()
x_max = tiles_df["x"].max()

y_min = tiles_df["y"].min()
y_max = tiles_df["y"].max()


# 5 × 4 = 20 spatial regions
x_bins = np.linspace(x_min, x_max, 6)
y_bins = np.linspace(y_min, y_max, 5)


selected = []

for i in range(5):

    for j in range(4):

        x_low = x_bins[i]
        x_high = x_bins[i + 1]

        y_low = y_bins[j]
        y_high = y_bins[j + 1]

        candidates = tiles_df[
            (tiles_df["x"] >= x_low) &
            (tiles_df["x"] <= x_high) &
            (tiles_df["y"] >= y_low) &
            (tiles_df["y"] <= y_high)
        ]

        if len(candidates) == 0:
            continue

        # Choose the tile closest to the center
        x_center = (x_low + x_high) / 2
        y_center = (y_low + y_high) / 2

        candidates = candidates.copy()

        candidates["distance"] = (
            (candidates["x"] - x_center) ** 2
            +
            (candidates["y"] - y_center) ** 2
        )

        best = candidates.sort_values(
            "distance"
        ).iloc[0]

        selected.append(best)


# --------------------------------------------------
# Final selection
# --------------------------------------------------

selected_df = pd.DataFrame(selected)

selected_df = selected_df.drop_duplicates(
    subset=["tile_id"]
)

selected_df = selected_df[
    ["tile_id", "x", "y"]
]

selected_df.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("\nSelected tiles:", len(selected_df))
print()

print(
    selected_df.to_string(index=False)
)

print(
    f"\nSaved to: {output_path}"
)