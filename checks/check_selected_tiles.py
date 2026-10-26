import pandas as pd
import os
import re

# --------------------------------------------------
# Selected tiles
# --------------------------------------------------

tiles_df = pd.read_csv("selected_tiles.csv")

tile_ids = tiles_df["tile_id"].astype(str).tolist()

# --------------------------------------------------
# Collections we want
# --------------------------------------------------

collections = [
    ("Atlanta_nadir7_catid_1030010003D22F00", 7),
    ("Atlanta_nadir10_catid_1030010003993E00", 10),
    ("Atlanta_nadir29_catid_1030010003315300", 29),
    ("Atlanta_nadir39_catid_1030010003832800", 39),
    ("Atlanta_nadir47_catid_10300100033C5200", 47),
    ("Atlanta_nadir53_catid_1030010003193D00", 53),
]

# --------------------------------------------------
# Check annotation CSVs
# --------------------------------------------------

results = []

for collection, angle in collections:

    csv_path = os.path.join(
        "summaryData",
        collection + "_Train.csv"
    )

    if not os.path.exists(csv_path):
        print("CSV NOT FOUND:", csv_path)
        continue

    df = pd.read_csv(csv_path)

    available_ids = set()

    for image_id in df["ImageId"].dropna().astype(str):

        match = re.search(r"_(\d+)_(\d+)$", image_id)

        if match:
            available_ids.add(
                f"{match.group(1)}_{match.group(2)}"
            )

    found = [
        tile for tile in tile_ids
        if tile in available_ids
    ]

    missing = [
        tile for tile in tile_ids
        if tile not in available_ids
    ]

    results.append({
        "angle": angle,
        "collection": collection,
        "available": len(found),
        "missing": len(missing)
    })

    print(
        f"{angle:>2}° | "
        f"Available: {len(found):>2} / {len(tile_ids)} | "
        f"Missing: {len(missing):>2}"
    )

    if missing:
        print("   Missing:", missing)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nSummary")
print("-" * 60)

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))