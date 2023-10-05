import random
import warnings
from typing import Any, Mapping, Sequence, Dict

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate as torch_default_collate


def pseudo_collate(data_batch: Sequence) -> Any:
    """将从数据集中采样的一批数据列表转换为训练可用格式. 
    与 pytorch 中的``default_collate``不同, "default_collate" 不会处理 "BaseDataElement".

    注意:
        ``default_collate``仅接受具有相同形状的输入张量. 

    Returns:
        Any: 与 ``data_batch`` 的 ``data_itement`` 格式相同的数据, 其中张量已经被堆叠, 并且ndarray、int、float已经被转换为张量. 
    """
    data_item = data_batch[0]
    data_item_type = type(data_item)

    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, "_fields"):
        return data_item_type(
            *(pseudo_collate(samples) for samples in zip(*data_batch))
        )
    elif isinstance(data_item, Sequence):
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                "each data_itement in list of batch should be of equal size"
            )
        try:
            return data_item_type([pseudo_collate(samples) for samples in data_batch])
        except TypeError:
            return [pseudo_collate(samples) for samples in data_batch]

    elif isinstance(data_item, Mapping):
        return data_item_type(
            {key: pseudo_collate([d[key] for d in data_batch]) for key in data_item}
        )

    else:
        return data_batch


def flatten_dict(d):
    """将字典中的非字典结构数据组合成列表
    注意: 仅对第一层开始的连续嵌套字典结构进行处理
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flattened_v = flatten_dict(v)
            for fk, fv in flattened_v.items():
                result[k + "." + fk] = fv
        else:
            result[k] = v
    return result


def combine_dicts(dict_list):
    """
    遍历 dict_list 中的所有数据, 将其中的非字典结构数据组合成列表
    """
    batch_dict = {}
    for d in dict_list:
        flattened_d = flatten_dict(d)
        for k, v in flattened_d.items():
            if k not in batch_dict:
                batch_dict[k] = []
            batch_dict[k].append(v)

    for k, v in batch_dict.items():
        batch_dict[k] = pseudo_collate(v)

    return batch_dict


def unflatten_dict(d):
    """
    将合并后的键重新展开为多层嵌套的字典结构
    """
    result = {}
    for key, value in d.items():
        keys = key.split(".")
        current_dict = result

        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]

        current_dict[keys[-1]] = value
    return result


def input_collate(data_batch: Sequence) -> Any:
    data_item = data_batch[0]
    assert "inputs" in data_item

    batch_dict = combine_dicts(data_batch)
    result = unflatten_dict(batch_dict)

    return result


sample_1 = {
    "inputs": {"audio_data": torch.randn(1, 10), "mfcc_data": torch.randn(1, 10, 13)},
    "data_samples": {"label_id": 0, "sample_rate": 10},
}


# 将3个样本堆叠在一起, 创建一个batch大小的Tensor对象
batch = [
    {
        "inputs": {
            "audio_data": torch.randn(1, 10),
            "mfcc_data": torch.randn(1, 10, 13),
            "text_data": [
                {"text": "hello world", "text_len": 11},
                {"text": "hello", "text_len": 5},
            ],
        },
        "data_samples": {"label_id": 0, "sample_rate": 10, "audio_path": "test0.wav"},
    },
    {
        "inputs": {
            "audio_data": torch.randn(1, 10),
            "mfcc_data": torch.randn(1, 10, 13),
            "text_data": [
                {"text": "hello world", "text_len": 11},
                {"text": "hello", "text_len": 5},
            ],
        },
        "data_samples": {"label_id": 1, "sample_rate": 10, "audio_path": "test1.wav"},
    },
    {
        "inputs": {
            "audio_data": torch.randn(1, 10),
            "mfcc_data": torch.randn(1, 10, 13),
            "text_data": [
                {"text": "hello world", "text_len": 11},
                {"text": "hello", "text_len": 5},
            ],
        },
        "data_samples": {"label_id": 2, "sample_rate": 10, "audio_path": "test2.wav"},
    },
]

batch = input_collate(batch)
print(batch)
pass
