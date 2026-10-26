import pandas as pd
import subprocess
import sys


# --------------------------------------------------
# Selected tiles
# --------------------------------------------------

tiles_df = pd.read_csv("selected_tiles.csv")

tile_ids = tiles_df["tile_id"].astype(str).tolist()


# --------------------------------------------------
# Collections
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
# Total work
# --------------------------------------------------

total = len(tile_ids) * len(collections)

completed = 0

print()
print("=" * 60)
print("SpaceNet dataset processing")
print("=" * 60)
print()

print("Tiles:", len(tile_ids))
print("Angles:", len(collections))
print("Total image/mask pairs:", total)
print()


# --------------------------------------------------
# Process every tile at every angle
# --------------------------------------------------

for tile_number, tile_id in enumerate(tile_ids, start=1):

    print()
    print(
        f"========== TILE {tile_number}/{len(tile_ids)}: "
        f"{tile_id} =========="
    )

    for collection, angle in collections:

        completed += 1

        print()
        print(
            f"[{completed}/{total}] "
            f"Processing {tile_id} at {angle}°"
        )

        result = subprocess.run(
            [
                sys.executable,
                "process_tile.py",
                collection,
                tile_id,
                str(angle)
            ]
        )

        if result.returncode != 0:

            print()
            print("ERROR processing:")
            print("Tile:", tile_id)
            print("Angle:", angle)
            print("Collection:", collection)

            print()
            print("Stopping so we can investigate.")

            sys.exit(1)


print()
print("=" * 60)
print("ALL PROCESSING COMPLETE")
print("=" * 60)
print()

print(f"Processed {completed} image/mask pairs.")