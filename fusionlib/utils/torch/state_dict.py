from numbers import Number
from typing import Any, Mapping, Union

from torch import nn

from .state_dict_arithmetic import *
from .type import _StateDict


class StateDict:
    """
    A wrapper class for PyTorch state dictionaries.

    This class allows arithmetic operations to be performed directly on state dictionaries. It can be initialized with a PyTorch module, a state dictionary, or another StateDict instance.

    Args:
        state_dict (Union[_StateDict, "StateDict", nn.Module]): The initial state dictionary. This can be a PyTorch module, a state dictionary, or another StateDict instance.

    Raises:
        TypeError: If the state_dict argument is not a PyTorch module, a state dictionary, or a StateDict instance.

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 10)
        >>> state_dict = StateDict(model)
        >>> print(state_dict['weight'])  # Access state dictionary items directly
        >>> state_dict2 = state_dict + 0.1  # Add a scalar to all parameters
        >>> state_dict3 = state_dict * 0.9  # Multiply all parameters by a scalar
    """

    def __init__(self, state_dict: Union[_StateDict, "StateDict", nn.Module]):
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        elif isinstance(state_dict, Mapping):
            self.state_dict = state_dict
        elif isinstance(state_dict, StateDict):
            self.state_dict = state_dict.state_dict
        else:
            raise TypeError(
                f"unsupported type for state_dict: '{type(state_dict).__name__}'"
            )

    def __add__(self, other: Union["StateDict", _StateDict, Number]) -> "StateDict":
        if isinstance(other, (StateDict, Mapping)):
            return StateDict(state_dict_add(self, other))
        elif isinstance(other, Number):
            return StateDict(state_dict_add_scalar(self, other))
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: 'StateDict' and '{type(other).__name__}'"
            )

    def __iadd__(self, other: Number) -> "StateDict":
        if isinstance(other, Number):
            for key in self.state_dict:
                self.state_dict[key] += other
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: 'StateDict' and '{type(other).__name__}'"
            )

    def __sub__(self, other: Union["StateDict", float]) -> "StateDict":
        if isinstance(other, (StateDict, Mapping)):
            return StateDict(state_dict_sub(self, other))
        elif isinstance(other, Number):
            return StateDict(state_dict_add_scalar(self, -other))
        else:
            raise TypeError(
                f"unsupported operand type(s) for -: 'StateDict' and '{type(other).__name__}'"
            )

    def __isub__(self, other: Number) -> "StateDict":
        if isinstance(other, Number):
            for key in self.state_dict:
                self.state_dict[key] -= other
        else:
            raise TypeError(
                f"unsupported operand type(s) for -: 'StateDict' and '{type(other).__name__}'"
            )

    def __mul__(self, other: Number) -> "StateDict":
        if isinstance(other, Number):
            return StateDict(state_dict_mul(self, other))
        else:
            raise TypeError(
                f"unsupported operand type(s) for *: 'StateDict' and '{type(other).__name__}'"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.state_dict, name)

    def to_vector(self) -> torch.Tensor:
        """
        Convert the state dictionary to a vector.

        Returns:
            torch.Tensor: The state dictionary as a vector.
        """
        return torch.nn.utils.parameters_to_vector(self.state_dict.values())

    def from_vector(self, vector: torch.Tensor):
        """
        Convert a vector to a state dictionary.

        Args:
            vector (torch.Tensor): The vector to convert to a state dictionary.
        """
        torch.nn.utils.vector_to_parameters(vector, self.state_dict.values())
        return self
