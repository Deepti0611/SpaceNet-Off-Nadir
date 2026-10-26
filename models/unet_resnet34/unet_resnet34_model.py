import torch
import segmentation_models_pytorch as smp


def get_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)

    print("Device:", device)
    print("Model: U-Net with ResNet34 encoder")
    print("Parameters:", sum(p.numel() for p in model.parameters()))