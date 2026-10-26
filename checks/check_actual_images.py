import subprocess
import re

collections = [
    ("Atlanta_nadir7_catid_1030010003D22F00", 7),
    ("Atlanta_nadir10_catid_1030010003993E00", 10),
    ("Atlanta_nadir29_catid_1030010003315300", 29),
    ("Atlanta_nadir39_catid_1030010003832800", 39),
    ("Atlanta_nadir47_catid_10300100033C5200", 47),
    ("Atlanta_nadir53_catid_1030010003193D00", 53),
]

# Get the 20 tiles that actually exist at 7°
collection7 = collections[0][0]

s3_path = (
    f"s3://spacenet-dataset/spacenet/SN4_buildings/sample/"
    f"{collection7}/Pan-Sharpen/"
)

result = subprocess.run(
    ["aws", "s3", "ls", "--no-sign-request", s3_path],
    capture_output=True,
    text=True
)

tiles = []

for line in result.stdout.splitlines():
    match = re.search(r"_(\d+_\d+)\.tif$", line)
    if match:
        tiles.append(match.group(1))

print("Tiles available at 7°:", len(tiles))
print()

# Check each tile at every angle
available_all = []

for tile in tiles:

    print(f"Checking {tile} ...", end=" ")

    exists_at_all = True

    for collection, angle in collections:

        filename = (
            f"Pan-Sharpen_{collection}_{tile}.tif"
        )

        path = (
            f"s3://spacenet-dataset/spacenet/SN4_buildings/sample/"
            f"{collection}/Pan-Sharpen/{filename}"
        )

        check = subprocess.run(
            ["aws", "s3", "ls", "--no-sign-request", path],
            capture_output=True,
            text=True
        )

        if not check.stdout.strip():
            exists_at_all = False
            break

    if exists_at_all:
        available_all.append(tile)
        print("YES")
    else:
        print("NO")

print()
print("=" * 60)
print("TILES AVAILABLE AT ALL SIX ANGLES")
print("=" * 60)

for tile in available_all:
    print(tile)

print()
print("Total:", len(available_all))