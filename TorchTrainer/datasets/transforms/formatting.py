from typing import Sequence, Union, List
from collections import defaultdict

import numpy as np
import torch

from .base import BaseTransform
from TorchTrainer.utils.registry import TRANSFORMS
from TorchTrainer.utils import is_str
from TorchTrainer.structures import ClsDataSample, LabelData


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
) -> torch.Tensor:
    """将各种支持的类型的对象转换为 `torch.Tensor` 类型."""

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        if isinstance(data[0], (torch.Tensor, np.ndarray)):
            return torch.stack([to_tensor(item) for item in data])
        elif isinstance(data[0], (int, float)):
            return torch.Tensor(data)
        else:
            return [to_tensor(item) for item in data]
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        return data


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):
    """将指定 key 或含 prefix 的数据转换为 `torch.Tensor` 类型."""

    def __init__(self, keys: Sequence[str] = [], prefix: str = None) -> None:
        self.keys = keys
        self.prefix = prefix if prefix is not None else ""

    def transform(self, results: dict) -> dict:
        if not self.keys:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        for key in self.keys:
            key_list = key.split(".")
            cur_item = results
            for i in range(len(key_list)):
                if key_list[i] not in cur_item:
                    raise KeyError(f"Can not find key {key}")
                if i == len(key_list) - 1:
                    cur_item[key_list[i]] = to_tensor(cur_item[key_list[i]])
                    break
                cur_item = cur_item[key_list[i]]

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(keys={self.keys})"


@TRANSFORMS.register_module()
class Transpose(BaseTransform):
    """将指定 key 的数据转置维度."""

    def __init__(self, order, keys: Sequence[str], prefix: str = None) -> None:
        self.keys = keys
        self.order = order
        self.prefix = prefix if prefix is not None else ""

    def transform(self, results: dict) -> dict:
        if not self.keys:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(keys={self.keys})"


@TRANSFORMS.register_module()
class ImageToTensor(BaseTransform):
    """根据指定 key 将图像类型数据转换为 `torch.Tensor` 类型.

    - 默认假设输入图像数据的维度为 (H, W, C), 并将其转换为 (C, H, W).
    - 如果图像数据的维度为 (H, W), 则转换为 (1, H, W).
    - 如果为指定 key, 则默认为含 "img" 的所有 key.

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys: dict = {}) -> None:
        self.keys = keys
        self.prefix = "img"

    def transform(self, results: dict) -> dict:
        if not self.keys:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = (to_tensor(img.transpose(2, 0, 1))).contiguous()
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(keys={self.keys})"


@TRANSFORMS.register_module()
class PackAudioClsInputs(BaseTransform):
    """为语音分类模型打包输入数据.
    需要提供的数据:
    inputs:
        - *_audio_data: np.ndarray
    meta:
        - audio_path: str
        - sample_rate: int
        - label_id: int
        - num_classes: int


    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]] = None,
        prefix: str = "audio_data",
    ) -> None:
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = prefix
        self.meta_keys = (
            "index",
            "audio_path",
            "sample_rate",
            "num_classes",
        )

    @staticmethod
    def format_input(input_):
        """用来将输入数据转换为 torch.Tensor 类型."""
        # if isinstance(input_, list):
        #     return [PackAudioClsInputs.format_input(item) for item in input_]
        if isinstance(input_, np.ndarray):
            if input_.ndim == 2:
                input_ = np.expand_dims(input_, 0)  # (H, W) -> (1, H, W)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_)  # (T, H, W)
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                input_ = to_tensor(input_).contiguous()
            else:
                input_ = to_tensor(input_)  # (T, C, H, W).
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f"Unsupported input type {type(input_)}.")
        return input_

    def transform(self, results: dict) -> dict:
        if self.keys is None:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        assert len(self.keys) > 0, "No audio data found."
        assert "label_id" in results, "label_id must be provided."

        packed_results = dict()
        packed_results["inputs"] = dict()

        for key in self.keys:
            packed_results["inputs"][key] = self.format_input(results.pop(key))
        data_sample = ClsDataSample()

        # Set default keys
        if "label_id" in results:
            data_sample.set_gt_label(results["label_id"])
            if "label_score" in results:
                data_sample.set_gt_score(results["label_score"])
            else:
                assert "num_classes" in results, "num_classes must be provided."
                data_sample.set_gt_score(
                    LabelData.label_to_onehot(
                        data_sample.gt_label.label, num_classes=results["num_classes"]
                    )
                )

        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(key, results[key], field_type="metainfo")

        packed_results["data_samples"] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        repr_str += f"(keys={self.keys})"
        repr_str += f"(prefix={self.prefix})"
        return repr_str


class PackVideoClsInputs(BaseTransform):
    """为视频分类模型打包输入数据.
    需要提供的数据:
    inputs:
        - *_video_data: np.ndarray
    meta:
        - video_path: str
        - fps: int
        - label_id: int
        - num_classes: int
    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]] = None,
        prefix: str = "video_data",
    ) -> None:
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = prefix
        self.meta_keys = (
            "index",
            "video_path",
            "fps",
            "num_classes",
        )

    @staticmethod
    def format_input(input_):
        """用来将输入数据转换为 torch.Tensor 类型."""
        if isinstance(input_, np.ndarray):
            if input_.ndim == 3:
                input_ = np.expand_dims(input_, 0)  # (H, W) -> (1, H, W)
            if input_.ndim == 4 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_)  # (T, H, W)
                input_ = to_tensor(input_)
            elif input_.ndim == 4:
                input_ = to_tensor(input_).contiguous()
            else:
                input_ = to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f"Unsupported input type {type(input_)}.")
        return input_

    def transform(self, results: dict) -> dict:
        if self.keys is None:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        assert len(self.keys) > 0, "No video data found."
        assert "label_id" in results, "label_id must be provided."

        packed_results = dict()
        packed_results["inputs"] = dict()

        for key in self.keys:
            packed_results["inputs"][key] = self.format_input(results.pop(key))
        data_sample = ClsDataSample()

        # Set default keys
        if "label_id" in results:
            data_sample.set_gt_label(results["label_id"])
            if "label_score" in results:
                data_sample.set_gt_score(results["label_score"])
            else:
                assert "num_classes" in results, "num_classes must be provided."
                data_sample.set_gt_score(
                    LabelData.label_to_onehot(
                        data_sample.gt_label.label, num_classes=results["num_classes"]
                    )
                )

        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(key, results[key], field_type="metainfo")

        packed_results["data_samples"] = data_sample
        return packed_results
