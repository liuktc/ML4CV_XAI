import torch
import torch.nn as nn
from captum.attr import DeepLiftShap
from util import cut_model_to_layer, cut_model_from_layer
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List
import numpy as np


class AttributionMethod:
    def __init__(self):
        pass

    def attribute(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        layer: str | nn.Module,
        target: torch.Tensor,
        baseline_dist: torch.Tensor = None,
    ):
        raise NotImplementedError()


class _DeepLiftShap(AttributionMethod):
    def attribute(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        layer: str | nn.Module,
        target: torch.Tensor,
        baseline_dist: torch.Tensor = None,
    ):
        model_to = cut_model_to_layer(
            model, layer, included=True
        )  # Model from the start up to the layer
        model_from = cut_model_from_layer(
            model, layer, included=False
        )  # Model from the layer to the end

        input_to_model = model_to(input_tensor)
        dl = DeepLiftShap(model_from)
        if baseline_dist is None:
            baseline_dist = torch.randn_like(input_to_model) * 0.001
        else:
            baseline_dist = model_to(baseline_dist)

        attributions, delta = dl.attribute(
            input_to_model, baseline_dist, target=target, return_convergence_delta=True
        )
        attributions = attributions.sum(dim=1, keepdim=True)

        return attributions


class _GradCAMPlusPlus(AttributionMethod):
    def attribute(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        layer: str | nn.Module,
        target: torch.Tensor,
        baseline_dist: torch.Tensor = None,
    ):
        cam = GradCAMPlusPlusNoResize(model=model, target_layers=[layer])
        targets = [ClassifierOutputTarget(t) for t in target]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = torch.Tensor(grayscale_cam)
        grayscale_cam = grayscale_cam.unsqueeze(1)
        return grayscale_cam


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
