import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import TwoSlopeNorm
from tqdm.auto import tqdm

from utils.masking import channel_mean_masking
from utils import cut_model_from_layer


def calculate_class_confidence(
    model: nn.Module,
    X: torch.Tensor,
    masks: torch.Tensor,
    device="cpu",
    return_probs=False,
) -> torch.Tensor:
    """
    Compute the confidence of the model for each class.
    """
    model.eval()
    with torch.no_grad():
        X = channel_mean_masking(X, masks)
        X.to(device)
        logits = model(X)
        if return_probs:
            return torch.nn.functional.softmax(logits, dim=1)
        else:
            return logits


def calculate_shap_values(
    model: nn.Module,
    to_explain: torch.Tensor,
    num_classes: int,
    n_samples: int = 200,
    device: str = "cpu",
    save_result: bool = False,
) -> torch.Tensor:
    """
    Compute SHAP values for a PyTorch model.
    When "removing" the contribution of a pixel, we replace it with the average value of that channel.

    Args:
        model (torch.nn.Module): The PyTorch model.
        X (torch.Tensor): The input image of shape (B, C, H, W).
        num_classes (int): The number of classes.
        n_samples (int): The number of samples to estimate the SHAP values.
        device (str): The device to run the model on.
    """
    to_explain = to_explain.squeeze(0).to(device)  # (C, H, W)
    H, W = to_explain.size(1), to_explain.size(2)
    num_pixels = H * W

    shap_values = torch.zeros(num_pixels, num_classes).to(device)

    for pixel in tqdm(range(num_pixels)):
        i, j = pixel // W, pixel % W

        masks_with_player = torch.ones(n_samples, H * W).to(device)
        masks_without_player = torch.ones(n_samples, H * W).to(device)
        for i in range(n_samples):
            # Randomly permute the pixels
            perm = torch.randperm(num_pixels).to(device)

            # Take a slice of perm from 0 to the pixel value
            pixel_index = (perm == pixel).nonzero()[0]

            # Set of player to consider
            perm_with_player = perm[: pixel_index + 1]
            perm_without_player = perm[:pixel_index]

            # Create the relative masks
            masks_with_player[i, perm_with_player] = 0
            masks_without_player[i, perm_without_player] = 0

        masks_with_player = masks_with_player.reshape(n_samples, H, W)
        masks_without_player = masks_without_player.reshape(n_samples, H, W)

        probs_with_player = calculate_class_confidence(
            model, to_explain, masks_with_player, device=device, return_probs=False
        )
        probs_without_player = calculate_class_confidence(
            model, to_explain, masks_without_player, device=device, return_probs=False
        )  # (nsamples, num_classes)

        # Compute the SHAP value
        shap_values[pixel] += (probs_with_player - probs_without_player).mean(dim=0)

    shap_values = shap_values / n_samples

    if save_result:
        save_shap_values(shap_values, ".", "all", n_samples)
    return shap_values


def calculate_shap_values_layer(
    model: nn.Module,
    X: torch.tensor,
    layer_name: str,
    num_classes: int,
    n_samples: int = 200,
    device: str = "cpu",
    save_result: bool = False,
) -> torch.Tensor:
    """
    Compute SHAP values for a PyTorch model.
    When "removing" the contribution of a pixel, we replace it with the average value of that channel.
    Given a layer, "cut" the model at that layer and compute the SHAP values for that activation map.

    Args:
        model (torch.nn.Module): The PyTorch model.
        X (torch.Tensor): The input image of shape (B, C, H, W).
        num_classes (int): The number of classes.
        n_samples (int): The number of samples to estimate the SHAP values.
        device (str): The device to run the model on.
    """
    outputs = {}

    # Define a hook to save the output of the specified layer
    def hook(module, input, output):
        outputs[layer_name] = output

    # Register the hook
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in the model.")
    hook_handle = layer.register_forward_hook(hook)

    # Perform a forward pass through the model
    with torch.no_grad():
        model(X)

    # Remove the hook
    hook_handle.remove()

    # Explain the output of the specified layer
    to_explain = outputs[layer_name]  # (B, C, H, W)

    # Cut the model at the specified layer
    model = cut_model_from_layer(model, layer_name)

    shap_values = calculate_shap_values(
        model, to_explain, num_classes, n_samples, device, save_result=False
    )

    if save_result:
        save_shap_values(shap_values, ".", layer_name, n_samples)

    return shap_values


def save_shap_values(
    shap_values: torch.Tensor, folder: str, layer_name: str, n_samples: int
):
    """
    Save the SHAP values to a file.
    """
    # Check if the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    path = os.path.join(folder, f"shap_values_{layer_name}_{n_samples}.pt")

    while os.path.exists(path):
        # Change the path name
        path = path.replace(".pt", "_1.pt")

    torch.save(shap_values, path)


def load_shap_values(folder: str, layer_name: str, n_samples: int) -> torch.Tensor:
    """
    Load the SHAP values from a file.
    """
    path = os.path.join(folder, f"shap_values_{layer_name}_{n_samples}.pt")
    return torch.load(path)


def plot_shap_values(original_image, shap_values, class_index):
    shap_values = shap_values[:, class_index]

    H = int(math.sqrt(shap_values.shape[0]))
    W = H
    shap_values = shap_values.reshape(H, W)

    # Upsample to original image size by bilinear interpolation
    up = nn.Upsample(original_image.shape[-2:], mode="bilinear")
    shap_values = up(shap_values.unsqueeze(0).unsqueeze(0)).squeeze().numpy()

    # Normalize with vcenter=0 to map 0 to white
    norm = TwoSlopeNorm(vmin=shap_values.min(), vcenter=0, vmax=shap_values.max())
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.squeeze().permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(shap_values, cmap="coolwarm", norm=norm)
    plt.colorbar()
    plt.show()
