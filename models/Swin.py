import timm
from timm.data import create_transform
import torch

MODEL_NAME = "swin_tiny_patch4_window7_224"

model = timm.create_model(MODEL_NAME, pretrained=True)

swin_preprocess = create_transform(
    input_size=model.default_cfg['input_size'],  # (3, 224, 224)
    mean=model.default_cfg['mean'],
    std=model.default_cfg['std'],
    interpolation=model.default_cfg['interpolation'],
    crop_pct=model.default_cfg['crop_pct']
)

def swin_imagenettewoof():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = torch.nn.Linear(768, 20)
    return model

def swin_imagenet():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    return model

def swin_PascalVOC():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = torch.nn.Linear(768, 20)
    return model

def swin_Syntetic():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = torch.nn.Linear(768, 6)
    return model

