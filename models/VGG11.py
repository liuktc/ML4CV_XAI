import torch.nn as nn
from torchvision.models import vgg11


def vgg11_PascalVOC():
    model = vgg11()
    model.classifier[-1] = nn.Linear(4096, 20)
    return model
