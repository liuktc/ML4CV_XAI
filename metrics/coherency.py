import torch
import torch.nn as nn
from .utils import mix_image_and_saliency, BaseMetric
from utils import AttributionMethod


def batch_pearson_coherency(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes Pearson correlation for a batch of matrices.

    Args:
        A (torch.Tensor): Tensor of shape (batch_size, m, n).
        B (torch.Tensor): Tensor of shape (batch_size, m, n).

    Returns:
        torch.Tensor: PCC for each matrix pair in the batch.
    """
    # Reshape to (batch_size, m*n)
    a = A.view(A.shape[0], -1)
    b = B.view(B.shape[0], -1)

    # Compute mean-centered matrices
    a_centered = a - a.mean(dim=1, keepdim=True)
    b_centered = b - b.mean(dim=1, keepdim=True)

    # Compute covariance and stds
    cov = (a_centered * b_centered).sum(dim=1) / (a.shape[1] - 1)
    std_a = torch.sqrt((a_centered**2).sum(dim=1) / (a.shape[1] - 1))
    std_b = torch.sqrt((b_centered**2).sum(dim=1) / (b.shape[1] - 1))

    # Avoid division by zero
    eps = 1e-8
    rho = cov / (std_a * std_b + eps)
    return rho


class Coherency(BaseMetric):
    def __init__(self):
        super().__init__("coherency")
        pass

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        labels: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        layer: nn.Module = None,
        **kwargs,
    ) -> torch.Tensor:
        # Coherency is defined as as the pearson correlation between the attribution on the image and the attribution on the image * saliency map
        mixed_images = mix_image_and_saliency(test_images, saliency_maps)

        mixed_attributions = attribution_method.attribute(
            input_tensor=mixed_images,
            model=model,
            layer=layer,
            target=labels,
        )

        print(mixed_images.shape, mixed_attributions.shape, saliency_maps.shape)

        # Compute the correlation between mixed_attributions and saliency_maps
        pearson = (batch_pearson_coherency(mixed_attributions, saliency_maps) + 1) / 2

        if return_mean:
            return pearson.mean()

        return pearson.item()
