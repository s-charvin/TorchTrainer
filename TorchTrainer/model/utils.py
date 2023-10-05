import logging
import warnings
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from TorchTrainer.utils.logging import print_log


def stack_batch(
    tensor_list: List[torch.Tensor],
    pad_size_divisor: int = 1,
    pad_value: Union[int, float] = 0,
) -> torch.Tensor:
    """将多个张量堆叠成一个批次, 并使用右下角填充模式对张量进行填充, 以达到最大形状.  如果``pad_size_divisor > 0``, 则添加填充以确保每个维度的形状可被``pad_size_divisor``整除.

    Args:
        tensor_list (List[Tensor]): 具有相同维度的张量列表.
        pad_size_divisor (int): 则添加填充以确保每个维度的形状可被 ``pad_size_divisor`` 整除.  Defaults to 1
        pad_value (int, float): 填充值.  Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list, list
    ), f"Expected input type to be list, but got {type(tensor_list)}"
    assert tensor_list, "`tensor_list` could not be an empty list"
    assert len({tensor.ndim for tensor in tensor_list}) == 1, (
        f"Expected the dimensions of all tensors must be the same, "
        f"but got {[tensor.ndim for tensor in tensor_list]}"
    )

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = (
        torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    )
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


def detect_anomalous_params(loss: torch.Tensor, model) -> None:
    parameters_in_graph = set()
    visited = set()

    def traverse(grad_fn):
        if grad_fn is None:
            return
        if grad_fn not in visited:
            visited.add(grad_fn)
            if hasattr(grad_fn, "variable"):
                parameters_in_graph.add(grad_fn.variable)
            parents = grad_fn.next_functions
            if parents is not None:
                for parent in parents:
                    grad_fn = parent[0]
                    traverse(grad_fn)

    traverse(loss.grad_fn)
    for n, p in model.named_parameters():
        if p not in parameters_in_graph and p.requires_grad:
            print_log(
                f"{n} with shape {p.size()} is not " f"in the computational graph \n",
                logger="current",
                level=logging.ERROR,
            )


def merge_dict(*args):
    """合并所有字典为一个字典"""
    output = dict()
    for item in args:
        assert isinstance(item, dict), (
            f"all arguments of merge_dict should be a dict, but got " f"{type(item)}"
        )
        output.update(item)
    return output


try:
    import torch.fx

    # make torch.fx skip trace `merge_dict`.
    merge_dict = torch.fx.wrap(merge_dict)

except ImportError:
    warnings.warn(
        "Cannot import torch.fx, `merge_dict` is a simple function "
        "to merge multiple dicts"
    )


class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    """一个没有输入维度检查的通用 BatchNorm 层, 为了方便转换 SyncBatchNorm"""

    def _check_input_dim(self, input: torch.Tensor):
        return


def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """辅助函数, 将模型中的所有 `SyncBatchNorm` 层转换为 `BatchNormXd` 层."""
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]

    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            # no_grad() may not be needed here but just to be consistent with `convert_sync_batchnorm()`
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        # qconfig exists in quantized models
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        # 当调用`add_module`时, 一些自定义模块或第三方实现的模块可能会引发错误. 因此, 请尝试捕获该错误并不要抛出它. 
        try:
            module_output.add_module(name, revert_sync_batchnorm(child))
        except Exception:
            print_log(
                f"Failed to convert {child} from SyncBN to BN!",
                logger="current",
                level=logging.WARNING,
            )
    del module
    return module_output


def convert_sync_batchnorm(module: nn.Module, implementation="torch") -> nn.Module:
    """辅助函数, 将模型中的所有 `BatchNorm` 层转换为 `SyncBatchNorm`."""
    module_output = module

    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if implementation == "torch":
            SyncBatchNorm = torch.nn.modules.batchnorm.SyncBatchNorm
        else:
            raise ValueError(
                'sync_bn should be "torch", but got ' f"{implementation}"
            )

        module_output = SyncBatchNorm(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )

        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child, implementation))
    del module
    return module_output
