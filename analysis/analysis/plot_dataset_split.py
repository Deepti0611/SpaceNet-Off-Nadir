import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("splits/final_dataset_split.csv")

split_counts = df["split"].value_counts()

# Keep the desired order
order = ["train", "validation", "test"]
split_counts = split_counts.reindex(order)

plt.figure(figsize=(7, 5))

plt.bar(
    split_counts.index,
    split_counts.values
)

plt.xlabel("Dataset Split")
plt.ylabel("Number of Geographic Tiles")
plt.title("Geographic Tile Distribution Across Dataset Splits")

plt.tight_layout()

plt.savefig(
    "results/dataset_split_distribution.png",
    dpi=300
)

plt.show()