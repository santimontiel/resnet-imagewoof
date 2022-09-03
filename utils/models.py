import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

def resnet50_imagewoof() -> torch.nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=2048, out_features=10, bias=True)
    )
    return model