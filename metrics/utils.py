import torch


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
