import pandas as pd
import os
from shapely import wkt

collections = [
    ("Atlanta_nadir7_catid_1030010003D22F00", 7),
    ("Atlanta_nadir10_catid_1030010003993E00", 10),
    ("Atlanta_nadir29_catid_1030010003315300", 29),
    ("Atlanta_nadir39_catid_1030010003832800", 39),
    ("Atlanta_nadir47_catid_10300100033C5200", 47),
    ("Atlanta_nadir53_catid_1030010003193D00", 53),
]

tiles = [
    "733601_3724689",
    "734051_3740889",
    "735401_3738639",
    "735401_3743139",
    "736751_3730539",
    "737201_3735039",
    "739001_3722439",
    "739451_3731889",
    "739451_3743589",
    "741251_3730089",
    "741251_3740439",
    "741251_3742239",
    "742601_3727389",
    "743501_3721539",
    "743951_3734589",
    "745301_3723339",
    "746201_3729189",
    "747101_3741339",
    "748451_3720189",
    "748451_3722889",
]

results = []

for collection, angle in collections:

    csv_path = os.path.join(
        "summaryData",
        f"{collection}_Train.csv"
    )

    df = pd.read_csv(csv_path)

    df["clean_image_id"] = (
        df["ImageId"]
        .astype(str)
        .str.replace("Pan-Sharpen_", "", regex=False)
    )

    for tile in tiles:

        image_id = f"{collection}_{tile}"

        tile_df = df[
            df["clean_image_id"] == image_id
        ]

        for _, row in tile_df.iterrows():

            polygon = wkt.loads(row["PolygonWKT_Pix"])

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            results.append({
                "tile": tile,
                "angle": angle,
                "building_id": row["BuildingId"],
                "area_pixels": polygon.area
            })

results_df = pd.DataFrame(results)

print("\nBUILDING AREA STATISTICS")
print("=" * 70)

print(
    results_df["area_pixels"].describe()
)

print("\n\nAREA STATISTICS BY ANGLE")
print("=" * 70)

print(
    results_df
    .groupby("angle")["area_pixels"]
    .agg([
        "count",
        "sum",
        "mean",
        "median",
        "min",
        "max"
    ])
)

print("\n\nAREA STATISTICS BY TILE")
print("=" * 70)

tile_stats = (
    results_df
    .groupby("tile")["area_pixels"]
    .agg([
        "count",
        "sum",
        "mean",
        "median",
        "min",
        "max"
    ])
)

print(tile_stats)

results_df.to_csv(
    "building_area_statistics.csv",
    index=False
)

tile_stats.to_csv(
    "building_area_by_tile.csv"
)

print("\nSaved:")
print("building_area_statistics.csv")
print("building_area_by_tile.csv")