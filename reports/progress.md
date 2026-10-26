| Step | Task                                                | Status                       |
| ---- | --------------------------------------------------- | ---------------------------- |
| 1    | Obtain SpaceNet 4 dataset                           | 🟢 Started / sample obtained |
| 2    | Understand dataset structure                        | 🟢 Done                      |
| 3    | Understand imagery                                  | 🟢 Done                      |
| 4    | Understand annotations                              | 🟢 Done                      |
| 5    | Generate building masks                             | 🟢 Done                      |
| 6    | Verify image–mask correspondence                    | 🟢 Done for several tiles    |
| 7    | Build reusable preprocessing pipeline               | 🟢 Done                      |
| 8    | Select a geographically diverse multi-angle dataset | 🟡 **Currently here**        |
| 9    | Analyze dataset statistically                       | ⏳ Next                       |
| 10   | Create train/validation/test split                  | ⏳                            |
| 11   | Select and justify pretrained models                | ⏳                            |
| 12   | Prepare model input pipeline                        | ⏳                            |
| 13   | Establish baseline models                           | ⏳                            |
| 14   | Evaluate by viewing angle                           | ⏳                            |
| 15   | Analyze failure cases                               | ⏳                            |

Original TIFF
┌─────────────┐
│ Band 1 = B  │
│ Band 2 = G  │
│ Band 3 = R  │
│ Band 4 = NIR│
└─────────────┘
       │
       │ select 3,2,1
       ↓
┌─────────────┐
│ R │ G │ B   │
└─────────────┘
       ↓
PyTorch tensor
[3, 900, 900]


Next: DataLoader

Now we'll wrap the Dataset in PyTorch's DataLoader.

The purpose is simply to take individual samples:

Dataset
  ↓
image 1 + mask 1
image 2 + mask 2
image 3 + mask 3
...

and group them into batches:

DataLoader
  ↓
Batch of images
[batch_size, 3, 900, 900]

Batch of masks
[batch_size, 1, 900, 900]

For your CPU machine, let's start with a small batch size of 2. We can change it later if necessary.


DATA PREPARATION
────────────────────────────────────

✓ SpaceNet imagery obtained
✓ Polygon annotations obtained
✓ Building masks generated
✓ 20 geographic tiles selected
✓ Six viewing angles verified
✓ 120 image instances created
✓ Geographic train/validation/test split
✓ Dataset integrity checked
✓ 4-band TIFFs preserved
✓ PyTorch installed
✓ PyTorch Dataset created
✓ RGB conversion implemented
✓ Binary masks implemented
✓ DataLoaders created
✓ Batch dimensions verified
✓ Image + mask visually verified

────────────────────────────────────
              DONE
────────────────────────────────────

84 train
18 validation
18 test
      ↓
PyTorch DataLoader
      ↓
RGB image: [3,900,900]
Mask:      [1,900,900]

PHASE 2 — MODEL SETUP

1. U-Net + ResNet34
2. DeepLabV3+ + ResNet50
3. SegFormer MiT-B0

C:\SpaceNet_Project
│
├── dataset/
│   ├── images/
│   ├── masks/
│   └── metadata.csv
│
├── pytorch_dataset.py
│
├── deeplabv3plus_resnet50_model.py
├── deeplabv3plus_resnet50_shape_test.py
├── deeplabv3plus_resnet50_train.py
├── deeplabv3plus_resnet50_best.pth
├── deeplabv3plus_resnet50_history.json
├── deeplabv3plus_resnet50_evaluate.py
└── deeplabv3plus_resnet50_test_results.json