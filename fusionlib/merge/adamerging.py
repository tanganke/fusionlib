from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(mod: nn.Module):
    """
    Converts a PyTorch module into a functional module by removing all parameters and returning their names.

    Args:
        mod (nn.Module): PyTorch module to be converted.

    Returns:
        Tuple[Tensor]: Tuple containing the original parameters of the module.
        List[str]: List containing the names of the removed parameters.
    """
    orig_params = tuple(mod.parameters())
    names: List[str] = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names


def load_weights(mod, names, params):
    """
    Loads weights into a PyTorch module.

    Args:
        mod (nn.Module): PyTorch module to load weights into.
        names (list): List of parameter names to load weights into.
        params (tuple): Tuple of weights to load into the module.
    """
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        """
        Initializes a wrapper for a PyTorch model.

        Args:
            model (nn.Module): PyTorch model to wrap.
            initial_weights (optional): Initial weights for the model. Defaults to None.
        """
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass of the wrapped PyTorch model.

        Args:
            images (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        features = self.model(images)
        return features


def softmax_entropy(x: Tensor):
    """
    Computes the softmax entropy of a tensor.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Softmax entropy of the input tensor.
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
