from .args import verify_str_arg
from .devices import num_devices
from .timer import timer

# if `torch` is available, import the `torch`-specific utilities
try:
    import torch

    from . import torch
except ImportError:
    pass
