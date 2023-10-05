from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from TorchTrainer.optim import OptimWrapper
from TorchTrainer.utils.registry import MODELS
from TorchTrainer.utils import is_list_of
from .base_module import BaseModule
from .data_preprocessor import BaseDataPreprocessor


class BaseModel(BaseModule):
    """模型基础类, 相比于结构基础类, 其实现了完整模型所需的基础功能, 包括:
        - 权重初始化(继承自 `BaseModule`)
        - batch 数据预处理
        - 损失解析(一般不需要重写)
        - 前向计算过程(需要重写, 并为多种模式: loss, predict, tensor 提供支持)
        - 训练过程(一般不需要重写)
        - 验证过程(一般不需要重写)
        - 测试过程(一般不需要重写)

    Attributes:
        data_preprocessor (:obj:`BaseDataPreprocessor`): 用于将输入的数据预处理为模型可接受的格式
        init_cfg (dict, optional): 模型参数初始化配置, 参考 `BaseModule`
    """

    def __init__(
        self,
        data_preprocessor: Optional[Union[dict, nn.Module]] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        if data_preprocessor is None:
            data_preprocessor = dict(type="BaseDataPreprocessor")
        if isinstance(data_preprocessor, nn.Module):
            self.data_preprocessor = data_preprocessor
        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            raise TypeError(
                "data_preprocessor should be a `dict` or "
                f"`nn.Module` instance, but got "
                f"{type(data_preprocessor)}"
            )

    def _run_forward(
        self, data: Union[dict, tuple, list], mode: str
    ) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks dict or list data for `forward` method and calls it."""
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError(
                "Output of `data_preprocessor` should be "
                f"list, tuple or dict, but got {type(data)}"
            )
        return results

    def train_step(
        self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper
    ) -> Dict[str, torch.Tensor]:
        """实现抽象的默认训练过程, 含预处理、前向传播、损失计算、优化和反向传播

        Args:
            data (dict or tuple or list): 采样自数据集的数据
            optim_wrapper (OptimWrapper): 优化器包装器
        Returns:
            Dict[str, torch.Tensor]: 用于日志记录的损失信息
        """
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode="loss")
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """

        Args:
            data (dict or tuple or list): 采样自数据集的数据
        Returns:
            一个列表, 包含了所有采样数据的预测结果
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode="predict")

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """

        Args:
            data (dict or tuple or list): 采样自数据集的数据

        Returns:
            一个列表, 包含了所有采样数据的预测结果
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode="predict")

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """解析网络的原始输出(损失)

        Args:
            losses (dict): 网络的原始输出(损失), 一般包含了多种损失和其他信息

        Returns:
            tuple[Tensor, dict]: 一个包含了损失和日志信息两个元素的元组, 用于日志记录和反向传播
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)

        return loss, log_vars

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = "tensor",
    ) -> Union[Dict[str, torch.Tensor], list]:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        Args:
            inputs (torch.Tensor): batch input tensor collated by :attr:`data_preprocessor`.
            data_samples (list, optional): data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and ``tensor``
                - ``loss``:  Return a ``dict`` used for backward and logging.
                - ``predict``: Return a list of results used for computing metric.
                - ``tensor``: return a data (such as tensor| tuple[tensor]| dict[tensor]) for custom use.
        """
        raise NotImplementedError

    def _set_device(self, device: torch.device) -> None:
        """为 `BaseDataPreprocessor` 实例递归设置设备."""

        def apply_fn(module):
            if not isinstance(module, BaseDataPreprocessor):
                return
            if device is not None:
                module._device = device

        self.apply(apply_fn)

    def to(self, *args, **kwargs) -> nn.Module:
        if args and isinstance(args[0], str) and "npu" in args[0]:
            args = tuple([list(args)[0].replace("npu", torch.npu.native_device)])
        if kwargs and "npu" in str(kwargs.get("device", "")):
            kwargs["device"] = kwargs["device"].replace("npu", torch.npu.native_device)

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._set_device(torch.device(device))
        return super().to(*args, **kwargs)

    def cuda(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> nn.Module:
        if device is None or isinstance(device, int):
            device = torch.device("cuda", index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def mlu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        device = torch.device("mlu", torch.mlu.current_device())
        self._set_device(device)
        return super().mlu()

    def npu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        device = torch.npu.current_device()
        self._set_device(device)
        return super().npu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        self._set_device(torch.device("cpu"))
        return super().cpu()
