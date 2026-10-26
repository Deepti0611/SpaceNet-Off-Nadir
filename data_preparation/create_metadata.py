import pandas as pd

data = [
    {
        "image_id": "743501_3721539",
        "angle": 7,
        "category": "nadir",
        "image_path": "images/Pan-Sharpen_Atlanta_nadir7_catid_1030010003D22F00_743501_3721539.tif",
        "mask_path": "masks/mask_7deg_743501_3721539.png"
    },
    {
        "image_id": "743501_3721539",
        "angle": 10,
        "category": "nadir",
        "image_path": "images/Pan-Sharpen_Atlanta_nadir10_catid_1030010003993E00_743501_3721539.tif",
        "mask_path": "masks/mask_743501_3721539.png"
    }
]

df = pd.DataFrame(data)

df.to_csv("dataset/metadata.csv", index=False)

print(df)
print("\nMetadata saved to dataset/metadata.csv")