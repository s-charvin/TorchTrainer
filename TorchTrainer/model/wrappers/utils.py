import torch.nn as nn

from TorchTrainer.utils.registry import MODEL_WRAPPERS, Registry


def is_model_wrapper(model: nn.Module, registry: Registry = MODEL_WRAPPERS):
    """检查一个模块是否是一个模型包装器, 例如 ``DataParallel``, ``DistributedDataParallel``, ``TTDataParallel``, ``TTDistributedDataParallel``."""
    module_wrappers = tuple(registry.module_dict.values())
    if isinstance(model, module_wrappers):
        return True

    if not registry.children:
        return False

    return any(is_model_wrapper(model, child) for child in registry.children.values())
