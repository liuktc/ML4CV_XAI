import torch.nn as nn
from collections import OrderedDict
import torch


def cut_model_from_layer(
    model: nn.Module,
    layer_name: str,
    included: bool = False,
    add_flatten_before_linear: bool = True,
) -> nn.Module:
    """
    Creates a new model that starts from the given layer of the original model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_name (str): The name of the layer from which to start the new model.
        included (bool): Whether to include the specified layer in the new model.
        add_flatten_before_linear (bool): Whether to add a flatten layer before the first linear layer.

    Returns:
        torch.nn.Module: A new model starting from the specified layer.
    """
    if isinstance(layer_name, nn.Module):
        layer_name = get_layer_name(model, layer_name)

    # Get all layers as an ordered dictionary
    modules = dict(model.named_modules())

    # Find the target layer
    if layer_name not in modules:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    # Create a new sequential model
    layers = OrderedDict()
    names = []
    layer_found = False
    flatten_added = False
    for name, module in model.named_modules():
        if name == layer_name:
            layer_found = True
        if layer_found:
            if name == layer_name and not included:
                continue
            else:
                if "." not in name:
                    continue

                if (
                    add_flatten_before_linear
                    and isinstance(module, nn.Linear)
                    and not flatten_added
                ):
                    layers["flatten"] = nn.Flatten()
                    flatten_added = True
                name = name.replace(".", "_")
                layers[name] = module
                names.append(name)

    if not layers or len(layers) == 0:
        raise ValueError(f"Could not cut the model from layer {layer_name}.")

    # Build and return the new model
    return nn.Sequential(layers)


def cut_model_to_layer(
    model: nn.Module,
    layer_name: str | nn.Module,
    included: bool = False,
) -> nn.Module:
    """
    Creates a new model that ends at the given layer of the original model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_name (str): The name of the layer at which to end the new model.
        included (bool): Whether to include the specified layer in the new model.

    Returns:
        torch.nn.Module: A new model ending at the specified layer.
    """
    if isinstance(layer_name, nn.Module):
        layer_name = get_layer_name(model, layer_name)

    # Get all layers as an ordered dictionary
    modules = dict(model.named_modules())

    # Find the target layer
    if layer_name not in modules:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    # Create a new sequential model
    layers = OrderedDict()
    names = []
    for name, module in model.named_modules():
        if name == layer_name:
            if included:
                name = name.replace(".", "_")
                layers[name] = module
            break
        else:
            if "." not in name:
                continue

            name = name.replace(".", "_")
            layers[name] = module
            names.append(name)

    if not layers or len(layers) == 0:
        raise ValueError(f"Could not cut the model to layer {layer_name}.")

    # Build and return the new model
    return nn.Sequential(layers)


def set_relu_inplace(model: nn.Module, inplace=False):
    """
    Sets the inplace parameter of all ReLU layers to False.
    This is needed for the DeepExplainer rom the shap library to work correctly.
    """
    for child in model.children():
        if isinstance(child, nn.ReLU):
            child.inplace = inplace
        else:
            set_relu_inplace(child)


def scale_single_tensor(y: torch.Tensor, perc: float = 0.5, tolerance: float = 1e-5):
    """
    Scale the input tensor so that the area under the curve is perc of the total area.
    y is assumed to be a 2D tensor whose values are in the range [0,1].

    Args:
        - y (torch.Tensor): 2D tensor whose values are in the range [0,1].
        - perc (float, optional): Percentage of the total area. Defaults to 0.5.
        - tolerance (float, optional): Tolerance on the result. Defaults to 1e-3.

    Returns:
        torch.Tensor: The scaled version of y.
    """
    # Scale y so that the area under the curve is perc of the total area
    assert 0 <= perc <= 1, "perc must be in the range [0,1]"
    assert 0 <= tolerance, "tolerance must be positive"
    assert y.min() == 0 and y.max() == 1, "y must be in the range [0,1]"
    y = y.clone().detach()

    H, W = y.shape
    TOT = H * W

    def integral(y):
        return torch.trapz(torch.trapz(y))

    def loss(y, perc):
        return (integral(y) - TOT * perc) ** 2

    alpha = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=0.1)

    for _ in range(10000):
        optimizer.zero_grad()
        loss_val = loss(y ** (alpha**2), perc)
        loss_val.backward()
        optimizer.step()
        if loss_val.item() < tolerance**2:
            break

    return y ** (alpha**2)


def scale_saliencies(
    saliencies: torch.Tensor, perc: float = 0.5, tolerance: float = 1e-5
):
    """
    Scale the input tensor so that the area under the curve is perc of the total area.
    y is assumed to be a 2D tensor whose values are in the range [0,1].

    Args:
        - y (torch.Tensor): 2D tensor whose values are in the range [0,1].
        - perc (float, optional): Percentage of the total area. Defaults to 0.5.
        - tolerance (float, optional): Tolerance on the result. Defaults to 1e-3.

    Returns:
        torch.Tensor: The scaled version of y.
    """
    # y.shape = (B, C, H, W)
    B, C, H, W = saliencies.shape

    assert C == 1, "Only single channel saliencies are supported"

    for b in range(B):
        saliencies[b, 0] = scale_single_tensor(saliencies[b, 0], perc, tolerance)

    return saliencies


def get_layer_name(model: nn.Module, layer: nn.Module):
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None
