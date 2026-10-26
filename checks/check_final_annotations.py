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

all_good = True

for collection, angle in collections:

    csv_path = os.path.join(
        "summaryData",
        f"{collection}_Train.csv"
    )

    df = pd.read_csv(csv_path)

    # Handle the occasional Pan-Sharpen_ prefix we observed
    df["clean_image_id"] = (
        df["ImageId"]
        .astype(str)
        .str.replace("Pan-Sharpen_", "", regex=False)
    )

    print(f"\n{angle}°")
    print("-" * 40)

    for tile in tiles:

        expected_id = f"{collection}_{tile}"

        count = (
            df["clean_image_id"] == expected_id
        ).sum()

        if count > 0:
            print(f"{tile} -> {count} buildings")
        else:
            print(f"{tile} -> MISSING")
            all_good = False

print("\n" + "=" * 60)

if all_good:
    print("ALL 20 TILES HAVE ANNOTATIONS AT ALL SIX ANGLES")
else:
    print("SOME ANNOTATIONS ARE MISSING")

print("=" * 60)