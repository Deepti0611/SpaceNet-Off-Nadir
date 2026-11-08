import torch
from transformers import SegformerForSemanticSegmentation


# ============================================================
# SEGFORMER-B0 MODEL
# ============================================================

def create_segformer():

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    return model


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SEGFORMER-B0 FOCAL LOSS MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = create_segformer().to(device)

    print("Model loaded successfully.")

    dummy_input = torch.randn(
        1, 3, 512, 512
    ).to(device)

    print("Input shape:", dummy_input.shape)

    with torch.no_grad():

        outputs = model(
            pixel_values=dummy_input
        )

    print("Output logits shape:", outputs.logits.shape)

    print("=" * 60)
    print("MODEL TEST COMPLETED")
    print("=" * 60)