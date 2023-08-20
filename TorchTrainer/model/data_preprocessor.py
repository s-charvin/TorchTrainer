import math
from typing import Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from TorchTrainer.utils.registry import MODELS
from TorchTrainer.structures import BaseDataElement
from TorchTrainer.utils import is_seq_of
from .utils import stack_batch

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str, None]


@MODELS.register_module()
class BaseDataPreprocessor(nn.Module):
    """模型的基础数据预处理结构, 用于整理和移动数据到目标设备(默认为 "cpu").

    Args:
        non_blocking (bool): 是否在将数据传输到设备时阻止当前进程
    Note:
        数据加载器返回的数据字典必须是一个字典,  并且至少包含 ``inputs`` 键.
    """

    def __init__(self, non_blocking: Optional[bool] = False):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device("cpu")

    def cast_data(self, data: CastData) -> CastData:
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, "_fields"):
            return type(data)(*(self.cast_data(sample) for sample in data))
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            return data.to(self.device, non_blocking=self._non_blocking)
        else:
            return data

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        return self.cast_data(data)

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.
        if args and isinstance(args[0], str) and "npu" in args[0]:
            args = tuple([list(args)[0].replace("npu", torch.npu.native_device)])
        if kwargs and "npu" in str(kwargs.get("device", "")):
            kwargs["device"] = kwargs["device"].replace("npu", torch.npu.native_device)

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.cuda.current_device())
        return super().cuda()

    def npu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.npu.current_device())
        return super().npu()

    def mlu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.mlu.current_device())
        return super().mlu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device("cpu")
        return super().cpu()


@MODELS.register_module()
class ImgDataPreprocessor(BaseDataPreprocessor):
    """图像数据预处理结构, 基于 ``BaseDataPreprocessor`` 并实现了以下功能:
        - 整理并将数据移动到目标设备.
        - 如果输入的形状为 (3, H, W), 则将输入从 bgr 转换为 rgb.
        - 使用定义的 std 和 mean 对图像进行归一化处理.
        - 使用定义的 ``pad_value`` 填充输入, 使其大小与当前批次的最大大小相同. 填充大小可以被定义的 ``pad_size_divisor`` 整除.
        - 将输入堆叠到 ``batch_inputs``.

    对于 "ImgDataPreprocessor", 单个输入的维度必须是(3, H, W).

    Args:
        mean (Sequence[float or int], optional): 图像通道的像素均值. 
            如果 ``bgr_to_rgb=True``, 则表示 R, G, B 通道的均值. 
            如果 `mean` 的长度为 1, 则表示所有通道具有相同的均值, 或输入为灰度图像. 
            如果未指定, 则不会对图像进行归一化. 默认为 None.
        std (Sequence[float or int], optional): 图像通道的像素标准差.
            同上. 默认为 None.
        pad_size_divisor (int): 填充后的图像大小应该可以被 ``pad_size_divisor`` 整除. 默认为 1.
        pad_value (float or int): 填充像素值. 默认为 0.
        bgr_to_rgb (bool): 是否将图像从 BGR 转换为 RGB. 默认为 False.
        rgb_to_bgr (bool): 是否将图像从 RGB 转换为 BGR. 默认为 False.
        non_blocking (bool): 是否在将数据传输到设备时阻止当前进程. 默认为 False.
    """

    def __init__(
        self,
        mean: Optional[Sequence[Union[float, int]]] = None,
        std: Optional[Sequence[Union[float, int]]] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        non_blocking: Optional[bool] = False,
    ):
        super().__init__(non_blocking)
        assert not (
            bgr_to_rgb and rgb_to_bgr
        ), "`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time"
        assert (mean is None) == (
            std is None
        ), "mean and std should be both None or tuple"
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                "`mean` should have 1 or 3 values, to be compatible with "
                f"RGB or gray image, but got {len(mean)} values"
            )
            assert len(std) == 3 or len(std) == 1, (
                "`std` should have 1 or 3 values, to be compatible with RGB "
                f"or gray image, but got {len(std)} values"
            )
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): 采样自数据集的数据, 必须包含 ``inputs`` 键.
            training (bool): 是否启用训练时间数据增强操作.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        _batch_inputs = data["inputs"]
        # Process data with `input_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim() == 3 and _batch_input.shape[0] == 3, (
                            "If the mean has 3 values, the input tensor "
                            "should in shape of (3, H, W), but got the tensor "
                            f"with shape {_batch_input.shape}"
                        )
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(
                batch_inputs, self.pad_size_divisor, self.pad_value
            )
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor "
                "or a list of tensor, but got a tensor with shape: "
                f"{_batch_inputs.shape}"
            )
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure efficiency
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(
                _batch_inputs, (0, pad_w, 0, pad_h), "constant", self.pad_value
            )
        else:
            raise TypeError(
                "Output of `cast_data` should be a dict of "
                "list/tuple with inputs and data_samples, "
                f"but got {type(data)}： {data}"
            )
        data["inputs"] = batch_inputs
        data.setdefault("data_samples", None)
        return data
