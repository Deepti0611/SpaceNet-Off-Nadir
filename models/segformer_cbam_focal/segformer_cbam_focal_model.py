
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


# ============================================================
# CBAM - CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(channels // reduction, 1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


# ============================================================
# CBAM - SPATIAL ATTENTION
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(self.conv(combined))

        return x * attention


# ============================================================
# COMPLETE CBAM MODULE
# ============================================================

class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


# ============================================================
# SEGFORMER-B0 + CBAM
# ============================================================

def create_segformer():

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    # Get the existing pretrained classifier
    original_classifier = model.decode_head.classifier

    # Obtain decoder feature-channel count from model config
    decoder_channels = model.config.decoder_hidden_size

    # Insert CBAM before the original classifier
    model.decode_head.classifier = nn.Sequential(
        CBAM(decoder_channels),
        original_classifier
    )

    return model


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SEGFORMER-B0 + CBAM + FOCAL LOSS MODEL TEST")
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
        outputs = model(pixel_values=dummy_input)

    print("Output logits shape:", outputs.logits.shape)

    attention_parameters = sum(
        p.numel()
        for p in model.decode_head.classifier[0].parameters()
    )

    print("CBAM parameters:", attention_parameters)

    print("=" * 60)
    print("MODEL TEST COMPLETED")
    print("=" * 60)