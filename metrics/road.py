from .utils import BaseMetric
from pytorch_grad_cam.metrics.road import (
    ROADCombined,
    ROADMostRelevantFirst,
    ROADLeastRelevantFirst,
)
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
        percentiles = [20, 40, 60, 80]
        road_combined = ROADCombined(percentiles=percentiles)
        targets = [ClassifierOutputSoftmaxTarget(i.item()) for i in class_idx]
        if len(saliency_maps.shape) == 4:
            saliency_maps = saliency_maps.squeeze(1)

        saliency_maps = saliency_maps.detach().cpu().numpy()
        scores = road_combined(test_images, saliency_maps, targets, model)

        if return_mean:
            scores = scores.mean()

        if "return_visualization" not in kwargs:
            return scores

        # Calculate visualization
        visualization_results = {}
        for perc in percentiles:
            for imputer in [
                ROADMostRelevantFirst(perc),
                ROADLeastRelevantFirst(perc),
            ]:
                scores, visualizations = imputer(
                    test_images,
                    saliency_maps,
                    targets,
                    model,
                    return_visualization=True,
                )

                if imputer.__class__.__name__ not in visualization_results:
                    visualization_results[imputer.__class__.__name__] = []

                visualization_results[imputer.__class__.__name__].append(
                    visualizations[0].detach().cpu()
                )

        return scores, visualization_results
