from contextlib import ExitStack, contextmanager
from typing import Dict, Union

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel

from TorchTrainer.utils.device import get_device
from TorchTrainer.optim import OptimWrapperDict
from TorchTrainer.utils.registry import MODEL_WRAPPERS
from .distributed import TTDistributedDataParallel


@MODEL_WRAPPERS.register_module()
class TTSeparateDistributedDataParallel(DistributedDataParallel):
    """分布式模型包装器, 可用于包含多个子模型的模型, 例如 GAN 等生成模型.
    - 使用 ``TTDistributedDataParallel`` 包装模型的子模型, 以实现子模型的分布式训练.
    - 调用子模块的 ``train_step``, ``val_step`` 和 ``test_step`` 方法, 以实现子模型的前向计算过程.
    """

    def __init__(
        self,
        module: nn.Module,
        broadcast_buffers: bool = False,
        find_unused_parameters: bool = False,
        **kwargs
    ):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        device = get_device()
        # Wrap the submodule with parameters of `self.module` to
        # `TTDistributedDataParallel`
        for name, sub_module in module._modules.items():
            # module without parameters.
            if next(sub_module.parameters(), None) is None:
                sub_module = sub_module.to(device)
            elif all(not p.requires_grad for p in sub_module.parameters()):
                sub_module = sub_module.to(device)
            else:
                sub_module = TTDistributedDataParallel(
                    module=sub_module.to(device),
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs
                )
            module._modules[name] = sub_module

    def train_step(
        self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapperDict
    ) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapperDict): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A dict of tensor for logging.
        """
        return self.module.train_step(data, optim_wrapper)

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.val_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the predictions of module during testing process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.test_step(data)

    @contextmanager
    def no_sync(self):
        """Enables ``no_sync`` context of all sub ``TTDistributedDataParallel``
        modules."""
        with ExitStack() as stack:
            for sub_ddp_model in self.module._modules.values():
                stack.enter_context(sub_ddp_model.no_sync())
                yield

    def train(self, mode: bool = True) -> "TTSeparateDistributedDataParallel":
        """Sets the module in training mode.

        In order to make the ddp wrapper inheritance hierarchy more uniform,
        ``SeparateDistributedDataParallel`` inherits from
        ``DistributedDataParallel``, but will not call its constructor.
        Since the attributes of ``DistributedDataParallel`` have not been
        initialized, call the ``train`` method of ``DistributedDataParallel``
        will raise an error if pytorch version <= 1.9. Therefore, override
        this method to call the ``train`` method of submodules.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                mode (``False``). Defaults to ``True``.

        Returns:
            Module: self.
        """
        self.training = mode
        self.module.train(mode)
        return self
