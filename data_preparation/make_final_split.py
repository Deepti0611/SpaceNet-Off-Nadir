import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Final verified tiles
# --------------------------------------------------

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

# --------------------------------------------------
# Building counts from previous analysis
# --------------------------------------------------

building_counts = {
    "733601_3724689": 43,
    "734051_3740889": 60,
    "735401_3738639": 119,
    "735401_3743139": 3,
    "736751_3730539": 14,
    "737201_3735039": 159,
    "739001_3722439": 1,
    "739451_3731889": 129,
    "739451_3743589": 95,
    "741251_3730089": 73,
    "741251_3740439": 35,
    "741251_3742239": 7,
    "742601_3727389": 47,
    "743501_3721539": 59,
    "743951_3734589": 118,
    "745301_3723339": 80,
    "746201_3729189": 2,
    "747101_3741339": 36,
    "748451_3720189": 31,
    "748451_3722889": 2,
}

# --------------------------------------------------
# Explicit geographic split
#
# We spread train/validation/test across
# different parts of the coordinate range.
# --------------------------------------------------

train_tiles = [
    "733601_3724689",
    "735401_3738639",
    "735401_3743139",
    "736751_3730539",
    "737201_3735039",
    "739451_3731889",
    "739451_3743589",
    "741251_3730089",
    "741251_3742239",
    "742601_3727389",
    "743501_3721539",
    "745301_3723339",
    "747101_3741339",
    "748451_3722889",
]

validation_tiles = [
    "734051_3740889",
    "741251_3740439",
    "748451_3720189",
]

test_tiles = [
    "739001_3722439",
    "743951_3734589",
    "746201_3729189",
]

# --------------------------------------------------
# Verify that every tile appears exactly once
# --------------------------------------------------

assert len(train_tiles) == 14
assert len(validation_tiles) == 3
assert len(test_tiles) == 3

assert len(
    set(train_tiles + validation_tiles + test_tiles)
) == 20

assert set(
    train_tiles + validation_tiles + test_tiles
) == set(tiles)

# --------------------------------------------------
# Create dataframe
# --------------------------------------------------

rows = []

for tile in tiles:

    x, y = tile.split("_")

    if tile in train_tiles:
        split = "train"

    elif tile in validation_tiles:
        split = "validation"

    else:
        split = "test"

    rows.append({
        "tile": tile,
        "x": int(x),
        "y": int(y),
        "building_count": building_counts[tile],
        "split": split
    })

df = pd.DataFrame(rows)

# --------------------------------------------------
# Print split statistics
# --------------------------------------------------

print("\nFINAL DATASET SPLIT")
print("=" * 70)

for split in ["train", "validation", "test"]:

    subset = df[df["split"] == split]

    print(f"\n{split.upper()}")
    print("-" * 50)

    print(
        subset[
            ["tile", "building_count"]
        ]
        .sort_values("building_count")
        .to_string(index=False)
    )

    print(
        f"\nTiles: {len(subset)}"
    )

    print(
        f"Buildings: {subset['building_count'].sum()}"
    )

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    "final_dataset_split.csv",
    index=False
)

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(11, 8))

for split in ["train", "validation", "test"]:

    subset = df[df["split"] == split]

    plt.scatter(
        subset["x"],
        subset["y"],
        s=120,
        label=split
    )

    for _, row in subset.iterrows():

        plt.annotate(
            row["tile"],
            (row["x"], row["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title(
    "Final Train / Validation / Test Geographic Split"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "final_dataset_split.png",
    dpi=200
)

plt.show()

print("\nSaved:")
print("final_dataset_split.csv")
print("final_dataset_split.png")