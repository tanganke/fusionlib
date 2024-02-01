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


def task_arithmetic_merge_state_dicts(
    pretrained_state_dict: _StateDict,
    finetuned_state_dicts: List[_StateDict],
    scaling_coef: float,
):
    task_vector = state_dict_sum(finetuned_state_dicts)
    task_vector = state_dict_mul(task_vector, scaling_coef)
    new_state_dict = state_dict_add(pretrained_state_dict, task_vector)
    return new_state_dict


@torch.no_grad()
def task_arithmetic_merge_modules(
    pretrained_model: nn.Module, finetuned_models: List[nn.Module], scaling_coef: float
):
    pretrained_state_dict = pretrained_model.state_dict()
    finetuned_state_dicts = [model.state_dict() for model in finetuned_models]
    new_state_dict = task_arithmetic_merge_state_dicts(
        pretrained_state_dict, finetuned_state_dicts, scaling_coef
    )
    model = deepcopy(pretrained_model)
    model.load_state_dict(new_state_dict)
    return model
