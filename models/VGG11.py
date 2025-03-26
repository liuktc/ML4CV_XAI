import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg11

vgg_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(  # Normalize using ImageNet mean and std
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def vgg11_PascalVOC():
    model = vgg11()
    model.classifier[-1] = nn.Linear(4096, 20)
    return model


def vgg11_Syntetic():
    model = vgg11()
    model.classifier[-1] = nn.Linear(4096, 6)
    return model


def vgg11_Imagenettewoof():
    model = vgg11()
    model.classifier[-1] = nn.Linear(4096, 20)
    return model
