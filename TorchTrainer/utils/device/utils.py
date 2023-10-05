import os
from typing import Optional

import torch

try:
    import torch_npu
    import torch_npu.npu.utils as npu_utils

    # Enable operator support for dynamic shape and
    # binary operator support on the NPU.
    npu_jit_compile = bool(os.getenv("NPUJITCompile", False))
    torch.npu.set_compile_mode(jit_compile=npu_jit_compile)
    IS_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:
    IS_NPU_AVAILABLE = False

try:
    import torch_dipu

    IS_DIPU_AVAILABLE = True
except Exception:
    IS_DIPU_AVAILABLE = False


def get_max_cuda_memory(device: Optional[torch.device] = None) -> int:
    """返回给定设备上张量占用的最大GPU内存 (以MB为单位) .
    Args:
        device (torch.device, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~torch.cuda.current_device`, if ``device`` is None.
            Defaults to None.

    Returns:
        int: The maximum GPU memory occupied by tensors in megabytes
        for a given device.
    """
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)], dtype=torch.int, device=device)
    torch.cuda.reset_peak_memory_stats()
    return int(mem_mb.item())


def is_cuda_available() -> bool:
    """如果CUDA PyTorch和CUDA设备存在, 则返回True,"""
    return torch.cuda.is_available()


def is_npu_available() -> bool:
    """如果存在Ascend PyTorch和npu设备, 则返回True."""
    return IS_NPU_AVAILABLE


def is_mlu_available() -> bool:
    """如果Cambricon PyTorch和mlu设备存在, 则返回True."""
    return hasattr(torch, "is_mlu_available") and torch.is_mlu_available()


def is_mps_available() -> bool:
    """如果存在mps设备, 则返回True."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_dipu_available() -> bool:
    """如果存在DIPU PyTorch和DIPU设备, 则返回True."""
    return IS_DIPU_AVAILABLE


def is_npu_support_full_precision() -> bool:
    """如果NPU设备支持全精度训练, 则返回True."""
    version_of_support_full_precision = 220
    return (
        IS_NPU_AVAILABLE
        and npu_utils.get_soc_version() >= version_of_support_full_precision
    )


DEVICE = "cpu"
if is_npu_available():
    DEVICE = "npu"
elif is_cuda_available():
    DEVICE = "cuda"
elif is_mlu_available():
    DEVICE = "mlu"
elif is_mps_available():
    DEVICE = "mps"
elif is_dipu_available():
    DEVICE = "dipu"


def get_device() -> str:
    """返回当前存在的设备类型, 包括cpu、npu、cuda、mlu、mps和dipu."""
    return DEVICE
