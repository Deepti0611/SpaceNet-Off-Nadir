import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.deeplabv3plus_resnet50.deeplabv3plus_resnet50_model import get_model

model = get_model()

total_params = 0
trainable_params = 0
frozen_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()

    if param.requires_grad:
        trainable_params += param.numel()
    else:
        frozen_params += param.numel()

print("\n========== PARAMETER STATUS ==========")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters:    {frozen_params:,}")

print("\n========== TRAINABLE PARAMETER GROUPS ==========")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

print("\n========== FROZEN PARAMETER GROUPS ==========")

for name, param in model.named_parameters():
    if not param.requires_grad:
        print(name)