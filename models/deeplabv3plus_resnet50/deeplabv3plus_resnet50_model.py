import torch
import segmentation_models_pytorch as smp


def get_model():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model


def set_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)

    print("Device:", device)
    print("Model: DeepLabV3+ with ResNet50 encoder")
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )