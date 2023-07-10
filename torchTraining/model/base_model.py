from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from torchTraining.optim import OptimWrapper
from torchTraining.utils.registry import MODELS
from torchTraining.utils import is_list_of
from .base_module import BaseModule
from .data_preprocessor import BaseDataPreprocessor


class BaseModel(BaseModule):
    """Base class for all algorithmic models.

    BaseModel implements the basic functions of the algorithmic model, such as
    weights initialize, batch inputs preprocess(see more information in
    :class:`BaseDataPreprocessor`), parse losses, and update model parameters.

    Subclasses inherit from BaseModel only need to implement the forward
    method, which implements the logic to calculate loss and predictions,
    then can be trained in the runner.

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.

    Attributes:
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`.
        init_cfg (dict, optional): Initialization config dict.
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
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
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
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode="predict")

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode="predict")

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

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
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

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
        """

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
        """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        if device is None or isinstance(device, int):
            device = torch.device("cuda", index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def mlu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.mlu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        device = torch.device("mlu", torch.mlu.current_device())
        self._set_device(device)
        return super().mlu()

    def npu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.npu`
        additionally.

        Returns:
            nn.Module: The model itself.

        Note:
            This generation of NPU(Ascend910) does not support
            the use of multiple cards in a single process,
            so the index here needs to be consistent with the default device
        """
        device = torch.npu.current_device()
        self._set_device(device)
        return super().npu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cpu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        self._set_device(torch.device("cpu"))
        return super().cpu()

    def _set_device(self, device: torch.device) -> None:
        """Recursively set device for `BaseDataPreprocessor` instance.

        Args:
            device (torch.device): the desired device of the parameters and
                buffers in this module.
        """

        def apply_fn(module):
            if not isinstance(module, BaseDataPreprocessor):
                return
            if device is not None:
                module._device = device

        self.apply(apply_fn)
