import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights


def resnet18_PascalVOC():
    model = resnet18(ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 20)
    return model


def resnet50_PascalVOC():
    model = resnet50(ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 20)
    return model
