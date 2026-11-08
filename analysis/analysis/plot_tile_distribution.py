import pandas as pd
import matplotlib.pyplot as plt

# Load our selected tiles
df = pd.read_csv("selected_tiles.csv")

plt.figure(figsize=(10, 8))

plt.scatter(
    df["x"],
    df["y"],
    s=100
)

for _, row in df.iterrows():
    plt.annotate(
        row["tile_id"],
        (row["x"], row["y"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Spatial Distribution of Selected SpaceNet Tiles")
plt.grid(True)

plt.tight_layout()
plt.show()