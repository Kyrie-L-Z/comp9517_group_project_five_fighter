import torch
import torch.nn as nn
from torchvision import models

def create_model(model_name="resnet18", num_classes=5, pretrained=True):
    """
    目前支持: resnet18 / mobilenet_v2
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unknown model_name: " + model_name)
    return model
