import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.deeplabv3plus_resnet50.deeplabv3plus_resnet50_model import get_model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return total, trainable, frozen


# ============================================================
# 1. LINEAR PROBING
# ============================================================

model = get_model()

for param in model.encoder.parameters():
    param.requires_grad = False

total, trainable, frozen = count_parameters(model)

print("\n========== LINEAR PROBING ==========")
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters:    {frozen:,}")


# ============================================================
# 2. PARTIAL FINE-TUNING
# ============================================================

model = get_model()

# Freeze the entire encoder first
for param in model.encoder.parameters():
    param.requires_grad = False

# Unfreeze later ResNet50 layers
for param in model.encoder.layer3.parameters():
    param.requires_grad = True

for param in model.encoder.layer4.parameters():
    param.requires_grad = True

# Decoder and segmentation head remain trainable

total, trainable, frozen = count_parameters(model)

print("\n========== PARTIAL FINE-TUNING ==========")
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters:    {frozen:,}")


# ============================================================
# 3. FULL FINE-TUNING
# ============================================================

model = get_model()

for param in model.parameters():
    param.requires_grad = True

total, trainable, frozen = count_parameters(model)

print("\n========== FULL FINE-TUNING ==========")
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters:    {frozen:,}")