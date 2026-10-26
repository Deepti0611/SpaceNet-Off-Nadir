import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


def create_segformer():
    """
    Create a pretrained SegFormer model for binary
    building segmentation.
    """

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    return model


if __name__ == "__main__":

    print("=" * 60)
    print("SEGFORMER MODEL TEST")
    print("=" * 60)

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Create model
    model = create_segformer()
    model = model.to(device)

    print("Model loaded successfully.")

    # --------------------------------------------------
    # Test input
    # Same format produced by pytorch_dataset.py:
    # [batch, channels, height, width]
    # --------------------------------------------------

    dummy_input = torch.randn(
        1, 3, 512, 512
    ).to(device)

    print("Input shape:", dummy_input.shape)

    # Forward pass
    with torch.no_grad():

        outputs = model(
            pixel_values=dummy_input
        )

    print("Output logits shape:", outputs.logits.shape)

    print("=" * 60)
    print("SEGFORMER TEST COMPLETED")
    print("=" * 60)