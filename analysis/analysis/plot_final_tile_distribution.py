import matplotlib.pyplot as plt

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

x = []
y = []

for tile in tiles:
    tile_x, tile_y = tile.split("_")
    x.append(int(tile_x))
    y.append(int(tile_y))

plt.figure(figsize=(10, 8))

plt.scatter(x, y, s=100)

for i, tile in enumerate(tiles):
    plt.annotate(
        tile,
        (x[i], y[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Spatial Distribution of Final Verified SpaceNet Tiles")
plt.grid(True)

plt.tight_layout()
plt.show()