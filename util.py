import torch.nn as nn
from collections import OrderedDict


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
    layer_name: str,
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


def set_relu_inplace(model, inplace=False):
    """
    Sets the inplace parameter of all ReLU layers to False.
    This is needed for the DeepExplainer rom the shap library to work correctly.
    """
    for child in model.children():
        if isinstance(child, nn.ReLU):
            child.inplace = inplace
        else:
            set_relu_inplace(child)
