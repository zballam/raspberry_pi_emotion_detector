import torch
from torchvision import models, transforms

from utility.tinycnn import TinyCNN


def build_model(model_name: str, num_classes: int):
    """
    Lightweight version of build_model just for inference on the Pi.
    """
    if model_name == "tinycnn":
        return TinyCNN(num_classes=num_classes)

    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_transforms():
    """
    Simple transforms for inference. Adjust to match how you trained.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])