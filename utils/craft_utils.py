import os

import torch
import torch.nn as nn
import numpy as np
from craft.craft_torch import Craft, torch_to_numpy
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def get_class_predictions_indices(
    dataset: Dataset,
    model: nn.Module,
    class_to_explain: int,
    device: str,
    path: str = "y_pred.pt",
):
    if os.path.exists(path):
        y_pred = torch.load(path)
    else:
        print("Calculating predictions for each image")
        y_pred = torch.tensor([]).to(device)

        dl = DataLoader(dataset, batch_size=32, shuffle=False)

        for x, _ in tqdm(dl, total=len(dl)):
            x = x.to(device)
            with torch.no_grad():
                outputs = model(x)
                predictions = torch.argmax(outputs, dim=1).view(-1)
                y_pred = torch.cat((y_pred, predictions), dim=0)

        print("Prediction calculation done, saving..")
        torch.save(y_pred, path)

    return torch.where(y_pred == class_to_explain)[0]


def calculate_craft_for_class(
    craft: Craft,
    images: torch.Tensor,
    class_to_explain: int,
):
    """Uses CRAFT to calculate the crops, crops_u, w, importances and images_u for a given class.

    Args:
        craft (Craft): CRAFT object already instantiated
        model (nn.Module): The model to evaluate
        dataset (Dataset): The *whole* dataset
        class_to_explain (int): The class to explain
        device (str): Torch device

    Returns:
        Tuple: crops, crops_u, w, importances, images_u
    """
    # image_indices = get_class_predictions_indices(
    #     dataset, model, class_to_explain, device
    # ).to(device)

    # class_images = torch.tensor([]).to(device)
    # for i in image_indices:
    #     x, _ = dataset[i]
    #     x = x.to(device)
    #     class_images = torch.cat((class_images, x.unsqueeze(0)), dim=0)

    # print(f"class_images.shape={class_images.shape}")

    # CRAFT will (1) create the patches, (2) find the concept
    # and (3) return the crops (crops), the embedding of the crops (crops_u), and the concept bank (w)
    crops, crops_u, w = craft.fit(images)
    crops = np.moveaxis(torch_to_numpy(crops), 1, -1)

    print(
        f"crops.shape={crops.shape}, crops_u.shape={crops_u.shape}, w.shape={w.shape}"
    )

    importances = craft.estimate_importance(images, class_id=class_to_explain)
    images_u = craft.transform(images)

    print(f"images_u.shape={images_u.shape}")

    return crops, crops_u, w, importances, images_u


def split_vgg(model: nn.Module):
    g = nn.Sequential(*list(model.children())[:-1])
    h = lambda x: model.classifier(torch.flatten(x, 1))

    return g, h


def split_resnet(model: nn.Module):
    g = nn.Sequential(*list(model.children())[:-1])
    h = lambda x: model.fc(x)

    return g, h
