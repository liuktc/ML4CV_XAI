import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .utils import BaseMetric
from utils import AttributionMethod


class ROC_AUC(BaseMetric):
    def __init__(self):
        super().__init__("roc_auc")

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
        if "mask" not in kwargs:
            raise ValueError("mask not provided in kwargs")
        mask = kwargs["mask"]
        mask = mask.detach().cpu().numpy()
        attribution = saliency_maps.detach().cpu().numpy()

        if mask.shape != attribution.shape:
            raise ValueError(
                f"mask and attribution shape mismatch, {mask.shape} != {attribution.shape}"
            )

        if len(mask.shape) != 4:
            raise ValueError(
                f"mask and attribution should have 4 dimensions, actual shape: {mask.shape}"
            )

        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(
                f"mask and attribution should have dimensions (1, 1, H, W), actual shape: {mask.shape}"
            )

        mask = mask.flatten()
        attribution = attribution.flatten()
        return roc_auc_score(mask, attribution)
