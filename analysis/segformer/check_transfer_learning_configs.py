
import os
import sys
import torch

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.insert(0, ROOT_DIR)

from models.segformer.segformer_model import create_segformer


# ---------------------------------------------------------
# PARAMETER COUNTING
# ---------------------------------------------------------

def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    return total, trainable, frozen


def print_parameters(model, title):

    total, trainable, frozen = count_parameters(model)

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")

    print("\nParameter groups:")

    for name, module in [
        ("Stage 0", model.segformer.stages[0]),
        ("Stage 1", model.segformer.stages[1]),
        ("Stage 2", model.segformer.stages[2]),
        ("Stage 3", model.segformer.stages[3]),
        ("Decode Head", model.decode_head)
    ]:

        params = list(module.parameters())

        group_total = sum(p.numel() for p in params)

        group_trainable = sum(
            p.numel() for p in params
            if p.requires_grad
        )

        group_frozen = group_total - group_trainable

        print(f"\n{name}")
        print(f"  Total:     {group_total:,}")
        print(f"  Trainable: {group_trainable:,}")
        print(f"  Frozen:    {group_frozen:,}")


# ---------------------------------------------------------
# 1. LINEAR PROBING
# ---------------------------------------------------------

print("\nLoading model for Linear Probing...")

model = create_segformer()

# Freeze the entire SegFormer encoder
for param in model.segformer.parameters():
    param.requires_grad = False

# Decode head remains trainable

print_parameters(model, "LINEAR PROBING")


# ---------------------------------------------------------
# 2. PARTIAL FINE-TUNING
# ---------------------------------------------------------

print("\nLoading model for Partial Fine-Tuning...")

model = create_segformer()

# Freeze the entire encoder first
for param in model.segformer.parameters():
    param.requires_grad = False

# Unfreeze the later encoder stages
for param in model.segformer.stages[2].parameters():
    param.requires_grad = True

for param in model.segformer.stages[3].parameters():
    param.requires_grad = True

# Decode head remains trainable

print_parameters(model, "PARTIAL FINE-TUNING")


# ---------------------------------------------------------
# 3. FULL FINE-TUNING
# ---------------------------------------------------------

print("\nLoading model for Full Fine-Tuning...")

model = create_segformer()

# Make all parameters trainable
for param in model.parameters():
    param.requires_grad = True

print_parameters(model, "FULL FINE-TUNING")


print("\n" + "=" * 65)
print("ALL CONFIGURATIONS CHECKED")
print("=" * 65)
