from collections import OrderedDict
from copy import deepcopy
from typing import List

import torch
from torch import Tensor, nn

from fusionlib.utils.torch import _StateDict
from fusionlib.utils.torch.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sum,
)

__all__ = ["task_arithmetic_merge_state_dicts", "task_arithmetic_merge_modules"]


def task_arithmetic_merge_state_dicts(
    pretrained_state_dict: _StateDict,
    finetuned_state_dicts: List[_StateDict],
    scaling_coef: float,
):
    """
    Examples:
        >>> pretrained_state_dict = model.state_dict()
        >>> finetuned_state_dicts = [model1.state_dict(), model2.state_dict()]
        >>> scaling_coef = 0.1
        >>> new_state_dict = task_arithmetic_merge_state_dicts(pretrained_state_dict, finetuned_state_dicts, scaling_coef)
    """
    task_vector = state_dict_sum(finetuned_state_dicts)
    task_vector = state_dict_mul(task_vector, scaling_coef)
    new_state_dict = state_dict_add(pretrained_state_dict, task_vector)
    return new_state_dict


@torch.no_grad()
def task_arithmetic_merge_modules(
    pretrained_model: nn.Module, finetuned_models: List[nn.Module], scaling_coef: float
):
    """
    Merges a pretrained model with a list of fine-tuned models using task arithmetic.

    Args:
        pretrained_model (nn.Module): The pretrained model.
        finetuned_models (List[nn.Module]): A list of fine-tuned models.
        scaling_coef (float): The scaling coefficient to apply to the sum of the fine-tuned state dictionaries.

    Returns:
        nn.Module: The merged model.

    Examples:
        >>> pretrained_model = torch.nn.Linear(10, 10)
        >>> finetuned_models = [torch.nn.Linear(10, 10), torch.nn.Linear(10, 10)]
        >>> scaling_coef = 0.1
        >>> new_model = task_arithmetic_merge_modules(pretrained_model, finetuned_models, scaling_coef)
    """
    pretrained_state_dict = pretrained_model.state_dict()
    finetuned_state_dicts = [model.state_dict() for model in finetuned_models]
    new_state_dict = task_arithmetic_merge_state_dicts(
        pretrained_state_dict, finetuned_state_dicts, scaling_coef
    )
    model = deepcopy(pretrained_model)
    model.load_state_dict(new_state_dict)
    return model
