import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


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
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg_out, max_out], dim=1)

        attention = self.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction=reduction
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class DeepLabV3PlusCBAM(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

        # DeepLabV3+ decoder produces 256 channels
        self.cbam = CBAM(channels=256)

    def forward(self, x):
        features = self.model.encoder(x)

        decoder_output = self.model.decoder(features)

        decoder_output = self.cbam(decoder_output)

        output = self.model.segmentation_head(decoder_output)

        return output

def set_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def get_model():
    return DeepLabV3PlusCBAM()


if __name__ == "__main__":
    model = get_model()

    print("Model: DeepLabV3+ with ResNet50 encoder + CBAM")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    # Dummy input test
    model.eval()

    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")