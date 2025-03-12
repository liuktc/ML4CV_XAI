from typing import List, Literal

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import numpy as np

from utils.attributions import AttributionMethod
from results import ResultMetrics
from utils import get_layer_name, scale_saliencies

from .average_drop import AverageDrop
from .increase_in_confidence import IncreaseInConfidence
from .insertion_curve import InsertionCurveAUC
from .deletion_curve import DeletionCurveAUC
from .utils import BaseMetric


import psutil
import os


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(
        f"Memory used: {process.memory_info().rss / 1024**2:.2f} MB"
    )  # Resident Set Size (RSS)


def calculate_metrics(
    model: nn.Module,
    attribute_method: AttributionMethod,
    test_dl: DataLoader,
    train_dl: DataLoader,
    layers: List[nn.Module],
    metrics: List[BaseMetric],
    result_metrics: ResultMetrics,
    upsample: nn.Module,
    device: torch.device | str = "cpu",
    rescale_saliency: bool = False,
    rescale_perc: float = 0.3,
    model_name: str = None,
    debug: bool = False,
) -> dict:
    """Function to calculate all the different metrics on the model using the given attribution method

    Args:
        model (nn.Module): The model to calculate the metrics on
        attribute_method (AttributionMethod): The attribution method to use
        test_dl (DataLoader): The dataloader to use for the test set
        train_dl (DataLoader): The dataloader to use for the train set (used only for the baseline distribution)
        layers (List[nn.Module]): The layers to calculate the metrics on
        upsample (nn.Module): The upsampling method to use. Either SimpleUpsampling or ERFUpsampling.
        device (torch.device | str, optional): The device to use. Defaults to "cpu".
        rescale_saliency (bool, optional): If set to true, rescale the saliency map to have a fixed are underneath it. Defaults to False.
        rescale_perc (float, optional): Parameter used to rescale the saliency map. Defaults to 0.3.

    Returns:
        dict: Dictionary containing the results of the metrics for each layer
    """
    # METRICS = ["avg_drop", "increase", "insertion_curve_AUC", "deletion_curve_AUC"]
    if model_name is None:
        model_name = model.__class__.__name__
    layer_names = {layer: get_layer_name(model, layer) for layer in layers}

    res = {
        layer_names[layer]: {metric.name: [] for metric in metrics} for layer in layers
    }

    # Use the train_dl as baseline distribution
    baseline_dist = torch.cat([images for images, _ in train_dl]).to(device)

    for layer in layers:
        for images, labels in tqdm(test_dl):
            if debug:
                print("-" * 80)
                print_memory_usage()

            labels = labels.to(device).reshape(-1)
            images = images.to(device)

            attributions = attribute_method.attribute(
                input_tensor=images,
                model=model,
                layer=layer,
                target=labels,
                baseline_dist=baseline_dist,
            )
            if debug:
                print_memory_usage()

            saliency_maps = upsample(attributions, images)

            if (
                torch.abs(
                    saliency_maps.amax(dim=(2, 3), keepdim=True)
                    - saliency_maps.amin(dim=(2, 3), keepdim=True)
                )
                < 1e-6
            ).any():
                print("A saliency map is constant, skipping batch")
                del images, labels, attributions, saliency_maps
                torch.cuda.empty_cache()
                continue

            saliency_maps = (
                saliency_maps - saliency_maps.amin(dim=(2, 3), keepdim=True)
            ) / (
                saliency_maps.amax(dim=(2, 3), keepdim=True)
                - saliency_maps.amin(dim=(2, 3), keepdim=True)
            )

            if saliency_maps.isnan().any():
                print("A saliency map is NaN, skipping batch")
                del images, labels, attributions, saliency_maps
                torch.cuda.empty_cache()
                continue

            if rescale_saliency:
                saliency_maps = scale_saliencies(saliency_maps, perc=rescale_perc)

            if debug:
                print_memory_usage()

            # Calculate all the metrics
            for metric in metrics:
                res[layer_names[layer]][metric.name].append(
                    metric(
                        model=model,
                        test_images=images,
                        saliency_maps=saliency_maps,
                        class_idx=labels,
                        attribution_method=attribute_method,
                        device=device,
                        baseline_dist=baseline_dist,
                        layer=layer,
                    )
                )

                if debug:
                    print_memory_usage()

            # avg_drop = (
            #     (
            #         average_drop(model, images, saliency_maps, labels, device)
            #         .detach()
            #         .cpu()
            #     )
            #     .mean()
            #     .item()
            # )

            # if debug:
            #     print_memory_usage()

            # increase = (
            #     (
            #         increase_in_confidence(model, images, saliency_maps, labels, device)
            #         .detach()
            #         .cpu()
            #     )
            #     .mean()
            #     .item()
            # )

            # if debug:
            #     print_memory_usage()

            # insertion_curve_AUC_score = (
            #     (
            #         insertion_curve_AUC(model, images, saliency_maps, labels, device)
            #         .detach()
            #         .cpu()
            #     )
            #     .mean()
            #     .item()
            # )

            # if debug:
            #     print_memory_usage()

            # deletion_curve_AUC_score = (
            #     (
            #         deletion_curve_AUC(model, images, saliency_maps, labels, device)
            #         .detach()
            #         .cpu()
            #     )
            #     .mean()
            #     .item()
            # )

            # if debug:
            #     print_memory_usage()

            # res[layer_names[layer]]["avg_drop"].append(avg_drop)
            # res[layer_names[layer]]["increase"].append(increase)
            # res[layer_names[layer]]["insertion_curve_AUC"].append(
            #     insertion_curve_AUC_score
            # )
            # res[layer_names[layer]]["deletion_curve_AUC"].append(
            #     deletion_curve_AUC_score
            # )

            # **Explicitly delete tensors and clear cache**
            del images, labels, attributions, saliency_maps
            torch.cuda.empty_cache()
            # del avg_drop, increase, insertion_curve_AUC_score, deletion_curve_AUC_score

        for metric in metrics:
            res[layer_names[layer]][metric.name] = np.mean(
                res[layer_names[layer]][metric.name]
            )

            result_metrics.add_result(
                model_name,
                attribute_method.__class__.__name__,
                layer_names[layer],
                metric.name,
                upsample.__class__.__name__,
                res[layer_names[layer]][metric.name],
            )

    return res
