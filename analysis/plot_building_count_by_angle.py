import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("analysis/dataset_building_statistics.csv")

angle_stats = (
    df.groupby("angle")["buildings"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    angle_stats["angle"].astype(str),
    angle_stats["buildings"]
)

plt.xlabel("Viewing Angle (degrees)")
plt.ylabel("Total Building Count")
plt.title("Building Count Distribution Across Viewing Angles")

plt.tight_layout()

plt.savefig(
    "results/building_count_by_angle.png",
    dpi=300
)

plt.show()