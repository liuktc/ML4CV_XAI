import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def average_drop(
    model: nn.Module,
    test_images: torch.Tensor,
    saliency_maps: torch.Tensor,
    class_idx: int,
) -> torch.Tensor:
    """
    The Average Drop refers to the maximum positive difference in the predictions made by the predictor using
    the input image and the prediction using the saliency map.
    Instead of giving to the model the original image, we give the saliency map as input and expect it to drop
    in performances if the saliency map doesn't contain relevant information.

    Parameters:

    model: torch.nn.Module
        The model to be evaluated.

    test_images: torch.Tensor
        The test images to be evaluated. Shape: (N, C, H, W)

    saliency_maps: torch.Tensor
        The saliency maps to be evaluated. Shape: (N, C, H, W)
    """

    test_images = test_images.to(model.device)
    saliency_maps = saliency_maps.to(model.device)

    test_preds = model(test_images)  # Shape: (N, num_classes)
    saliency_preds = model(saliency_maps)  # Shape: (N, num_classes)

    # Select only the relevant class
    test_preds = test_preds[:, class_idx]  # Shape: (N,)
    saliency_preds = saliency_preds[:, class_idx]  # Shape: (N,)

    numerator = test_preds - saliency_preds
    numerator[numerator < 0] = 0

    denominator = test_preds

    return torch.sum(numerator / denominator) * 100


def increase_in_confidence(
    model: nn.Module,
    test_images: torch.Tensor,
    saliency_maps: torch.Tensor,
    class_idx: int,
) -> torch.Tensor:
    """
    The number of times in the entire dataset that the model's confidence increased when providing only
    the saliency map as input.

    Parameters:

    model: torch.nn.Module
        The model to be evaluated.

    test_images: torch.Tensor
        The test images to be evaluated. Shape: (N, C, H, W)

    saliency_maps: torch.Tensor
        The saliency maps to be evaluated. Shape: (N, C, H, W)
    """

    test_images = test_images.to(model.device)
    saliency_maps = saliency_maps.to(model.device)

    test_preds = model(test_images)  # Shape: (N, num_classes)
    saliency_preds = model(saliency_maps)  # Shape: (N, num_classes)

    # Select only the relevant class
    test_preds = test_preds[:, class_idx]  # Shape: (N,)
    saliency_preds = saliency_preds[:, class_idx]  # Shape: (N,)

    numerator = test_preds - saliency_preds
    numerator[numerator < 0] = 1
    numerator[numerator > 0] = 0

    denominator = len(test_preds)  # N

    return torch.sum(numerator / denominator) * 100
