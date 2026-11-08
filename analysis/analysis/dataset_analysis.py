import pandas as pd
import os

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
    "raw_data",
    "summaryData",
    f"{collection}_Train.csv"
)

    df = pd.read_csv(csv_path)

    # Remove occasional Pan-Sharpen_ prefix
    df["clean_image_id"] = (
        df["ImageId"]
        .astype(str)
        .str.replace("Pan-Sharpen_", "", regex=False)
    )

    for tile in tiles:

        image_id = f"{collection}_{tile}"

        count = (
            df["clean_image_id"] == image_id
        ).sum()

        results.append({
            "tile": tile,
            "angle": angle,
            "buildings": count
        })

results_df = pd.DataFrame(results)

print("\nBUILDING COUNT BY ANGLE")
print("=" * 60)

print(
    results_df
    .groupby("angle")["buildings"]
    .agg(["count", "sum", "mean", "median", "min", "max"])
)

print("\n\nBUILDING COUNT BY TILE")
print("=" * 60)

table = results_df.pivot(
    index="tile",
    columns="angle",
    values="buildings"
)

print(table)

results_df.to_csv(
    "dataset_building_statistics.csv",
    index=False
)

table.to_csv(
    "building_counts_by_tile.csv"
)

print("\nStatistics saved:")
print("dataset_building_statistics.csv")
print("building_counts_by_tile.csv")