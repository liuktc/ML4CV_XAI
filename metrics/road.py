from .utils import BaseMetric
from pytorch_grad_cam.metrics.road import ROADCombined
import torch.nn as nn
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from utils import AttributionMethod


class RoadCombined(BaseMetric):
    def __init__(self):
        super().__init__("road_combined")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        road_combined = ROADCombined(percentiles=[20, 40, 60, 80])
        targets = [ClassifierOutputSoftmaxTarget(i.item()) for i in class_idx]
        if len(saliency_maps.shape) == 4:
            saliency_maps = saliency_maps.squeeze(1)

        saliency_maps = saliency_maps.detach().cpu().numpy()
        scores = road_combined(test_images, saliency_maps, targets, model)

        if return_mean:
            scores = scores.mean()

        return scores
