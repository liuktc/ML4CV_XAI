from typing import List, Literal

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torcheval.metrics.aggregation.auc import AUC
from torchvision.transforms.functional import gaussian_blur
from tqdm.auto import tqdm

from attributions import AttributionMethod
from results import ResultMetrics
from utils import get_layer_name, scale_saliencies


def average_drop(
    model: nn.Module,
    test_images: torch.Tensor,
    saliency_maps: torch.Tensor,
    class_idx: int | torch.Tensor,
    device: torch.device | str = "cpu",
    apply_softmax: bool = True,
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

    class_idx: int | torch.Tensor
        If int: the index of the class to be evaluated, the same for all the input images.
        if torch.Tensor: the index of the class to be evaluated for each input image. Shape: (N,)
    """
    test_images = test_images.to(device)
    # plt.imshow(test_images[0].permute(1, 2, 0).detach().cpu().numpy())
    # plt.show()

    saliency_maps = saliency_maps.to(device)
    # plt.imshow(saliency_maps[0].permute(1, 2, 0).detach().cpu().numpy())
    # plt.show()

    saliency_images = mix_image_and_saliency(test_images, saliency_maps)
    # plt.imshow(saliency_images[0].permute(1, 2, 0).detach().cpu().numpy())
    # plt.show()

    test_preds = model(test_images)  # Shape: (N, num_classes)
    saliency_preds = model(saliency_images)  # Shape: (N, num_classes)

    if apply_softmax:
        test_preds = nn.functional.softmax(test_preds, dim=1)
        saliency_preds = nn.functional.softmax(saliency_preds, dim=1)

    # Select only the relevant class
    if isinstance(class_idx, int):
        test_preds = test_preds[:, class_idx]  # Shape: (N,)
        saliency_preds = saliency_preds[:, class_idx]  # Shape: (N,)
    elif isinstance(class_idx, torch.Tensor):
        test_preds = test_preds[torch.arange(test_preds.size(0)), class_idx]
        saliency_preds = saliency_preds[torch.arange(saliency_preds.size(0)), class_idx]
    else:
        raise ValueError("class_idx should be either an int or a torch.Tensor")

    numerator = test_preds - saliency_preds
    numerator[numerator < 0] = 0

    denominator = test_preds

    return torch.sum(numerator / denominator) * 100


def increase_in_confidence(
    model: nn.Module,
    test_images: torch.Tensor,
    saliency_maps: torch.Tensor,
    class_idx: int | torch.Tensor,
    device: torch.device | str = "cpu",
    apply_softmax: bool = True,
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

    class_idx: int | torch.Tensor
        If int: the index of the class to be evaluated, the same for all the input images.
        if torch.Tensor: the index of the class to be evaluated for each input image. Shape (N,)
    """

    test_images = test_images.to(device)
    saliency_maps = saliency_maps.to(device)
    saliency_images = mix_image_and_saliency(test_images, saliency_maps)

    test_preds = model(test_images)  # Shape: (N, num_classes)
    saliency_preds = model(saliency_images)  # Shape: (N, num_classes)

    if apply_softmax:
        test_preds = nn.functional.softmax(test_preds, dim=1)
        saliency_preds = nn.functional.softmax(saliency_preds, dim=1)

    # Select only the relevant class
    if isinstance(class_idx, int):
        test_preds = test_preds[:, class_idx]  # Shape: (N,)
        saliency_preds = saliency_preds[:, class_idx]  # Shape: (N,)
    elif isinstance(class_idx, torch.Tensor):
        test_preds = test_preds[torch.arange(test_preds.size(0)), class_idx]
        saliency_preds = saliency_preds[torch.arange(saliency_preds.size(0)), class_idx]
    else:
        raise ValueError("class_idx should be either an int or a torch.Tensor")

    numerator = test_preds - saliency_preds
    numerator[numerator > 0] = 0
    numerator[numerator < 0] = 1

    denominator = len(test_preds)  # N

    return torch.sum(numerator / denominator) * 100


def mix_image_and_saliency(
    image: torch.Tensor, saliency_map: torch.Tensor
) -> torch.Tensor:
    """
    Mixes the image and the saliency map to create a new image.

    Parameters:

    image: torch.Tensor
        The input image. Shape: (B, C, H, W)

    saliency_map: torch.Tensor
        The saliency map. Shape: (B, C, H, W)
        Each element of the saliency map should be between 0 and 1.
    """
    # assert saliency_map.max() == 1 and saliency_map.min() == 0
    if saliency_map.max() != 1 or saliency_map.min() != 0:
        print("Saliency map should be normalized between 0 and 1")
        print(saliency_map.max(), saliency_map.min())
        raise ValueError
    return image * saliency_map


def deletion_curve(
    model: nn.Module,
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
    apply_softmax: bool = True,
    num_points: int = 30,
):
    """Generate the deletion curve as defined in https://arxiv.org/abs/1806.07421

    Args:
        model (nn.Module): The model to be evaluated
        image (torch.Tensor): The input image. Shape: (B, C, H, W)
        saliency_map (torch.Tensor): The saliency map. Shape: (B, C, H, W)
        device (torch.device | str, optional): The device to be used. Defaults to "cpu".
        labels (torch.Tensor): The labels of the input images. Shape: (B,)
        apply_softmax (bool, optional): Whether to apply softmax to the output. Defaults to True.
    """
    assert saliency_maps.shape[1] == 1, "Saliency map should be single channel"
    saliency_maps = saliency_maps.squeeze(1)

    B, C, H, W = images.shape
    num_pixels = H * W

    deletion_ranges = torch.zeros(B, num_points)
    deletion_values = torch.zeros(B, num_points)
    for b in range(B):
        image = images[b].unsqueeze(0)  # Shape: (1, C, H, W)
        # image = image.unsqueeze(0)  # Shape: (1, C, H, W)
        saliency_map = saliency_maps[b]  # Shape: (H, W)

        sm_flatten = saliency_map.flatten()
        best_indices = sm_flatten.argsort().flip(
            0
        )  # Indices of the saliency map sorted in descending order

        pixel_removed_perc = torch.linspace(0, 1, num_points)
        res = torch.zeros_like(pixel_removed_perc)

        # for i, perc in tqdm(enumerate(pixel_removed_perc)):
        for i, perc in enumerate(pixel_removed_perc):
            num_pixels_to_remove = int(num_pixels * perc)

            pixels_to_be_removed = best_indices[:num_pixels_to_remove]

            new_image = image.clone()
            new_image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W] = (
                0  # Remove the pixel by setting it to a constant value
            )

            new_image = new_image.to(device)

            # Compute the prediction confidence on the class_idx
            with torch.no_grad():
                preds = model(new_image)[0]
                if apply_softmax:
                    preds = nn.functional.softmax(preds, dim=0)[labels[b]]
                res[i] = preds

        deletion_ranges[b] = pixel_removed_perc
        deletion_values[b] = res

    return deletion_ranges, deletion_values


def insertion_curve(
    model: nn.Module,
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
    apply_softmax: bool = True,
    num_points: int = 30,
):
    """Generate the insertion curve as defined in https://arxiv.org/abs/1806.07421

    Args:
        model (nn.Module): The model to be evaluated
        image (torch.Tensor): The input image. Shape: (C, H, W)
        saliency_map (torch.Tensor): The saliency map. Shape: (H, W)
    """
    assert saliency_maps.shape[1] == 1, "Saliency map should be single channel"
    saliency_maps = saliency_maps.squeeze(1)

    B, C, H, W = images.shape
    num_pixels = H * W

    insertion_ranges = torch.zeros(B, num_points)
    insertion_values = torch.zeros(B, num_points)

    for b in range(B):
        image = images[b].unsqueeze(0)  # Shape: (1, C, H, W)
        saliency_map = saliency_maps[b]  # Shape: (H, W)
        # Apply gaussian filter to the image
        # Values taken from https://github.com/eclique/RISE/blob/master/Evaluation.ipynb
        kernel_size = 11
        sigma = 5
        blurred_image = gaussian_blur(image, kernel_size, sigma)

        sm_flatten = saliency_map.flatten()
        best_indices = sm_flatten.argsort().flip(
            0
        )  # Indices of the saliency map sorted in descending order

        pixel_removed_perc = torch.linspace(0, 1, num_points)
        res = torch.zeros_like(pixel_removed_perc)

        # plt.figure(figsize=(20, 20))
        # for i, perc in tqdm(enumerate(pixel_removed_perc)):
        for i, perc in enumerate(pixel_removed_perc):
            # plt.subplot(6, 5, i + 1)
            num_pixels_to_remove = int(num_pixels * perc)

            pixels_to_be_removed = best_indices[:num_pixels_to_remove]

            new_image = blurred_image.clone()
            # new_image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W] = (
            #     0  # Remove the pixel by setting it to a constant value
            # )
            new_image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W] = (
                image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W]
            )

            new_image = new_image.to(device)

            # plt.imshow(new_image[0].permute(1, 2, 0).detach().cpu().numpy())
            # print(new_image[0].max(), new_image[0].min())

            # Compute the prediction confidence on the class_idx
            with torch.no_grad():
                preds = model(new_image)[0]
                if apply_softmax:
                    preds = nn.functional.softmax(preds, dim=0)[labels[b]]
                res[i] = preds

        insertion_ranges[b] = pixel_removed_perc
        insertion_values[b] = res

    return insertion_ranges, insertion_values


def deletion_curve_AUC(
    model: nn.Module,
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
    apply_softmax: bool = True,
):
    B, C, H, W = images.shape
    ins_range, insertion = deletion_curve(
        model, images, saliency_maps, labels, device, apply_softmax
    )
    res = torch.zeros(B)
    for i in range(B):
        insertion_auc = AUC()
        insertion_auc.update(ins_range[i], insertion[i])
        res[i] = insertion_auc.compute()

    return res


def insertion_curve_AUC(
    model: nn.Module,
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
    apply_softmax: bool = True,
):
    B, C, H, W = images.shape
    ins_range, insertion = insertion_curve(
        model, images, saliency_maps, labels, device, apply_softmax
    )
    res = torch.zeros(B)
    for i in range(B):
        insertion_auc = AUC()
        insertion_auc.update(ins_range[i], insertion[i])
        res[i] = insertion_auc.compute()

    return res


def calculate_metrics(
    model: nn.Module,
    attribute_method: AttributionMethod,
    test_dl: DataLoader,
    train_dl: DataLoader,
    layers: List[nn.Module],
    result_metrics: ResultMetrics,
    device: torch.device | str = "cpu",
    upsample: Literal[
        "nearest", "linear", "bilinear", "bicubic", "trilinear"
    ] = "nearest",
    rescale_saliency: bool = False,
    rescale_perc: float = 0.3,
    model_name: str = None,
) -> dict:
    """Function to calculate all the different metrics on the model using the given attribution method

    Args:
        model (nn.Module): The model to calculate the metrics on
        attribute_method (AttributionMethod): The attribution method to use
        test_dl (DataLoader): The dataloader to use for the test set
        train_dl (DataLoader): The dataloader to use for the train set (used only for the baseline distribution)
        layers (List[nn.Module]): The layers to calculate the metrics on
        upsample (Literal[&#39;nearest&#39;, &#39;linear&#39;, &#39;bilinear&#39;, &#39;bicubic&#39;, &#39;trilinear&#39;], optional): Upsampling method to upsample the saliency map. Defaults to "nearest".
        rescale_saliency (bool, optional): If set to true, rescale the saliency map to have a fixed are underneath it. Defaults to False.
        rescale_perc (float, optional): Parameter used to rescale the saliency map. Defaults to 0.3.

    Returns:
        dict: Dictionary containing the results of the metrics for each layer
    """
    if model_name is None:
        model_name = model.__class__.__name__
    layer_names = {layer: get_layer_name(model, layer) for layer in layers}

    res = {
        layer_names[layer]: {
            "avg_drop": [],
            "increase": [],
            "insertion_curve_AUC": [],
            "deletion_curve_AUC": [],
        }
        for layer in layers
    }

    for layer in layers:
        for images, labels in tqdm(test_dl):
            labels = labels.to(device).reshape(-1)
            images = images.to(device)

            # Use the train_dl as baseline distribution
            baseline_dist = torch.cat([images for images, _ in train_dl])

            attributions = attribute_method.attribute(
                model,
                input_tensor=images,
                layer=layer,
                target=labels,
                baseline_dist=baseline_dist,
            )

            up = nn.Upsample(size=images.shape[2:], mode=upsample)
            saliency_maps = up(attributions)
            saliency_maps = (
                saliency_maps - saliency_maps.amin(dim=(2, 3), keepdim=True)
            ) / (
                saliency_maps.amax(dim=(2, 3), keepdim=True)
                - saliency_maps.amin(dim=(2, 3), keepdim=True)
            )

            if rescale_saliency:
                saliency_maps = scale_saliencies(saliency_maps, perc=rescale_perc)

            avg_drop = average_drop(model, images, saliency_maps, labels)
            increase = increase_in_confidence(model, images, saliency_maps, labels)
            insertion_curve_AUC_score = insertion_curve_AUC(
                model, images, saliency_maps, labels
            )
            deletion_curve_AUC_score = deletion_curve_AUC(
                model, images, saliency_maps, labels
            )

            res[layer_names[layer]]["avg_drop"].append(avg_drop)
            res[layer_names[layer]]["increase"].append(increase)
            res[layer_names[layer]]["insertion_curve_AUC"].append(
                insertion_curve_AUC_score
            )
            res[layer_names[layer]]["deletion_curve_AUC"].append(
                deletion_curve_AUC_score
            )

        # For each layer, average the results
        res[layer_names[layer]]["avg_drop"] = torch.mean(
            torch.stack(res[layer_names[layer]]["avg_drop"])
        ).item()
        res[layer_names[layer]]["increase"] = torch.mean(
            torch.stack(res[layer_names[layer]]["increase"])
        ).item()
        res[layer_names[layer]]["insertion_curve_AUC"] = torch.mean(
            torch.stack(res[layer_names[layer]]["insertion_curve_AUC"])
        ).item()
        res[layer_names[layer]]["deletion_curve_AUC"] = torch.mean(
            torch.stack(res[layer_names[layer]]["deletion_curve_AUC"])
        ).item()

        result_metrics.add_results_all_layers(
            model_name, attribute_method.__class__.__name__, res
        )

    return res
