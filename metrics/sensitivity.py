from .utils import BaseMetric
import torch
import torch.nn as nn
import numpy as np
from captum.metrics import sensitivity_max
from utils import AttributionMethod


class Sensitivity(BaseMetric):
    def __init__(self):
        super().__init__("sensitivity")

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
        # **kwargs needs to contain baseline_dist and layer

        def attribution_wrapper(images, model, layer, targets, baseline_dist):
            if type(images) is tuple and len(images) == 1:
                images = images[0]

            BATCH_SIZE = 2

            res = []
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i : i + BATCH_SIZE]
                print(batch.shape)
                res.append(
                    attribution_method.attribute(
                        batch, model, layer, targets, baseline_dist
                    ).detach()
                )

            return torch.cat(res, dim=0)

        # Set the **kwargs to contain model, layer, targets, baseline_dist
        kwargs["model"] = model
        kwargs["targets"] = class_idx

        sens = sensitivity_max(attribution_wrapper, test_images.clone(), **kwargs)
        if return_mean:
            sens = torch.mean(sens)

        return sens.detach().cpu()
