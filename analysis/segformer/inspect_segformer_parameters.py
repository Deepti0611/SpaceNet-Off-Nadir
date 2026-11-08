
import os
import sys
import torch

# Add project root to Python path
ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.insert(0, ROOT_DIR)

from models.segformer.segformer_model import create_segformer


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    frozen = total - trainable

    return total, trainable, frozen


def main():

    print("=" * 70)
    print("SEGFORMER PARAMETER INSPECTION")
    print("=" * 70)

    model = create_segformer()

    # 1. Overall parameter counts
    total, trainable, frozen = count_parameters(model)

    print("\n========== PARAMETER COUNTS ==========")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")

    # 2. Top-level modules
    print("\n========== TOP-LEVEL MODULES ==========")

    for name, module in model.named_children():
        print(f"\n{name}: {type(module).__name__}")

    # 3. Encoder structure
    print("\n========== ENCODER MODULES ==========")

    print("\n========== SEGFORMER INTERNAL STRUCTURE ==========")

    for name, module in model.segformer.named_children():
        print(f"\n{name}: {type(module).__name__}")

    for child_name, child_module in module.named_children():
        print(f"    {child_name}: {type(child_module).__name__}")

        if isinstance(module, torch.nn.ModuleList):
            print(f"Number of stages: {len(module)}")

            for i, stage in enumerate(module):
                print(f"  Stage {i}: {type(stage).__name__}")

    # 4. Parameter groups
    print("\n========== PARAMETER GROUPS ==========")

    groups = {}

    for name, param in model.named_parameters():
        prefix = name.split(".")[0]

        if prefix not in groups:
            groups[prefix] = {
                "total": 0,
                "trainable": 0
            }

        groups[prefix]["total"] += param.numel()

        if param.requires_grad:
            groups[prefix]["trainable"] += param.numel()

    for name, values in groups.items():
        frozen = values["total"] - values["trainable"]

        print(f"\n{name}")
        print(f"  Total:     {values['total']:,}")
        print(f"  Trainable: {values['trainable']:,}")
        print(f"  Frozen:    {frozen:,}")

    # 5. Detailed encoder parameter names
    print("\n========== ENCODER PARAMETER NAMES ==========")

    for name, param in model.segformer.named_parameters():
        print(f"{name}: {param.numel():,}")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
