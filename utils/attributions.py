import torch
import torch.nn as nn
from captum.attr import DeepLiftShap
from .util import (
    cut_model_to_layer,
    cut_model_from_layer,
    min_max_normalize,
    calculate_erf,
    post_process_erf,
    calculate_erf_on_attribution,
)
from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List, Callable
import numpy as np

import psutil
import os


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(
        f"Memory used: {process.memory_info().rss / 1024**2:.2f} MB"
    )  # Resident Set Size (RSS)


class AttributionMethod:
    def __init__(self, name: str = None):
        self.name = name
        pass

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: str | nn.Module,
        target: torch.Tensor,
        baseline_dist: torch.Tensor = None,
    ):
        raise NotImplementedError()


class SimpleUpsampling(nn.Module):
    def __init__(self, size, mode="bilinear", name: str = "bilinearUpsampling"):
        super().__init__()
        self.size = size
        self.mode = mode
        self.up = nn.Upsample(size=self.size, mode=self.mode)
        self.name = name

    def forward(self, attribution, **kwargs):
        return self.up(attribution)


class ERFUpsampling(nn.Module):
    """
    Upsampling using effective receptive field (ERF) upsampling.
    """

    def __init__(
        self,
        model,
        layer: nn.Module,
        device="cpu",
        post_process_filter: Callable = post_process_erf,
    ):
        super().__init__()
        # self.model = model
        self.device = device
        self.model = cut_model_to_layer(model, layer, included=True)
        self.post_process_filter = post_process_filter

    def forward(self, attribution: torch.Tensor, image):
        erf = calculate_erf(self.model, image, device=self.device)

        # Rescale the attribution map using the ERF values
        result = np.zeros(erf.shape[2:], dtype=np.float32)
        assert attribution.shape[0] == 1 and attribution.shape[1] == 1
        attribution = attribution[0][0].detach().cpu().numpy()
        attribution = attribution.astype(np.float32)

        for i in range(erf.shape[0]):
            for j in range(erf.shape[1]):
                result += attribution[i, j] * erf[i, j]

        # Sum over channels
        result = result.sum(axis=0, keepdims=True)

        result = torch.Tensor(result).unsqueeze(0)

        result = min_max_normalize(result)

        if self.post_process_filter is not None:
            result = self.post_process_filter(result)

        return torch.Tensor(result).to(self.device)


def get_layer_name(model: nn.Module, layer: nn.Module):
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None


class ERFUpsamplingFast(nn.Module):
    def __init__(
        self,
        post_process_filter: Callable = post_process_erf,
        name: str = "ERFUpsamplingFast",
    ):
        super().__init__()
        self.post_process_filter = post_process_filter
        self.name = name

    def forward(self, attribution: torch.Tensor, image, device, model, layer, **kwargs):
        self.model = cut_model_to_layer(model, layer, included=True)
        self.layer_number = int(get_layer_name(model, layer).split(".")[1])
        # erf = calculate_erf(self.model, image, device=self.device)
        result = calculate_erf_on_attribution(self.model, image, attribution, device)

        result = result.to(device=device)
        result = result.sum(axis=0).unsqueeze(0).unsqueeze(0)

        result = min_max_normalize(result)

        if self.post_process_filter is not None:
            result = self.post_process_filter(result, self.layer_number)

        return torch.Tensor(result).to(device)


class _DeepLiftShap(AttributionMethod):
    def __init__(
        self, baseline_images: torch.Tensor = None, name: str = "DeepLiftShap"
    ):
        super().__init__(name=name)
        self.baseline_images = baseline_images

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: str | nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        model_to = cut_model_to_layer(
            model, layer, included=True
        )  # Model from the start up to the layer
        model_from = cut_model_from_layer(
            model, layer, included=False
        )  # Model from the layer to the end

        input_to_model = model_to(input_tensor)
        dl = DeepLiftShap(model_from)
        if self.baseline_images is None:
            baseline_dist = torch.randn_like(input_to_model) * 0.001
        else:
            baseline_dist = model_to(self.baseline_images)

        attributions, delta = dl.attribute(
            input_to_model, baseline_dist, target=target, return_convergence_delta=True
        )
        attributions = attributions.sum(dim=1, keepdim=True)

        # METTI ABS e poi normalizzi
        if normalize:
            attributions = min_max_normalize(attributions)

        return attributions


class _ScoreCAM(AttributionMethod):
    def __init__(
        self,
        name: str = "ScoreCAM",
    ):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: str | nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = ScoreCAMNoResize(model=model, target_layers=[layer])

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute ScoreCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor


class ScoreCAMNoResize(ScoreCAM):
    """Just the ScoreCAM but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class _GradCAMPlusPlus(AttributionMethod):
    def __init__(
        self,
        name: str = "GradCAMPlusPlus",
    ):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: str | nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = GradCAMPlusPlusNoResize(model=model, target_layers=[layer])

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute GradCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor


class GradCAMPlusPlusNoResize(GradCAMPlusPlus):
    """Just the GradCAMPlusPlus but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class EigenCAMNoResize(EigenCAM):
    """Just the EigenCAM but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class _EigenCAM(AttributionMethod):
    def __init__(
        self,
        name: str = "EigenCAM",
    ):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: str | nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = EigenCAMNoResize(model=model, target_layers=[layer])

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute EigenCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor


class LayerCAMNoResize(LayerCAM):
    """Just the LayerCAM but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class _LayerCAM(AttributionMethod):
    def __init__(self, name: str = "LayerCAM"):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: str | nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = LayerCAMNoResize(model=model, target_layers=[layer])

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute LayerCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor
