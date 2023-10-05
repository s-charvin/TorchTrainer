import torch

from ..version_utils import digit_version
from .parrots_wrapper import TORCH_VERSION

_torch_version_meshgrid_indexing = "parrots" not in TORCH_VERSION and digit_version(
    TORCH_VERSION
) >= digit_version("1.10.0a0")


def torch_meshgrid(*tensors):
    """一个torch.meshgrid的包装器, 用于兼容不同版本的PyTorch.
    自从 PyTorch 1.10.0a0, torch.meshgrid 支持参数 ``indexing``.
    此包装器避免在使用高版本PyTorch时出现警告, 并避免在使用旧版本 PyTorch 时出现兼容性问题.
    """
    if _torch_version_meshgrid_indexing:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)  # Uses indexing='ij' by default
