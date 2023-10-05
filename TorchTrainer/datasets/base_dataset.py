import os.path as osp
import copy
import gc
import pickle
import numpy as np
import re
from typing import Callable, List, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset


from TorchTrainer.utils.fileio import load, isfile, dump
from TorchTrainer.utils.registry import TRANSFORMS


class Transform:
    """组合数据处理函数(transform)"""

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(
                        f"transform should be a callable object, "
                        f"but got {type(transform)}"
                    )
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f"transform must be a callable object or dict, "
                    f"but got {type(transform)}"
                )

    def __call__(self, data: dict) -> Optional[dict]:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class BaseDataset(Dataset):
    """数据集基类, 所有数据集都应该继承该类.
    - 支持根据数据集路径(root)从本地数据库加载数据集信息(metainfo)和数据集列表(data_list).
    - 支持通过过滤规则(filter)过滤数据集, 四种过滤规则("replace", "dropna", "contains", "query").
    - 支持序列化数数据集, 以节省内存.
    - 支持通过索引访问数据集中的数据, 同时经过处理函数(transforms)对数据进行处理.
    - 支持获取数据集中指定索引位置处未处理的数据.
    - 支持获取数据集子集(subset).


    data_list: List[dict] = [
        {"img": "a.jpg", "label": "cat", label_id: 0},
        {"img": "b.jpg", "label": "dog", label_id: 1},
        {"img": "c.png", "label": "cat", label_id: 0},
        {"img": "d.png", "label": "dog", label_id: 1},
        ...
    ]
    metainfo: dict = {
        "classes": ["cat", "dog", ...],
        "num_classes": 2,
    }

    filter: dict = {

        "replace": {"label": ("cat", "dog")},
        # "replace": {"label": {"cat": "dog"}},
        # "replace": {"label": [("cat", "dog"), ("dog", "cat")]},
        # "replace": {"label": [{"cat": "dog"}, {"dog": "cat"}]},

        "drop": {"label": "cat"},
        # "drop": {"label": ["cat", "dog"]},
        # "drop": "label",
        # "drop": ["label", "img"],

        "contain": {"img": ".jpg"},
        # "contain": {"img": [".jpg", ".png"]},

        "query": "label_id == 0",
        # "query": "label_id != 0",
        # "query": "label_id > 0",
        # "query": "label_id >= 0",
        # "query": "label_id < 0",
        # "query": "label_id <= 0",
        # "query": "label_id in [0, 1]",
        # "query": "label_id not in [0, 1]",
        # "query": "label_id > 0 & label_id < 2",
        # "query": "label_id < 0 | label_id > 2",
    }
    """

    def __init__(
        self,
        root: Optional[str],
        filter: Optional[dict] = None,
        serialize: bool = True,
        transforms: List[Union[dict, Callable]] = [],
    ):
        super().__init__()

        self.root = root
        self.serialize = serialize
        self.filter_cfg = copy.deepcopy(filter)
        self.metainfo = {}
        self.data_list: List[dict] = []
        self.data_address: np.ndarray = np.array([])
        self.data_bytes: np.ndarray = np.array([])

        if isfile(self.root):
            self.load_info_from_local()
        else:
            self.data_list, self.metainfo = self.load_info()
        self.filter()
        if self.serialize:
            self.data_bytes, self.data_address = self._serialize_data()

        self.pipeline = Transform(transforms)

    def load_info_from_local(self):
        data = load(self.root)
        if "data_list" not in data or "metainfo" not in data:
            raise ValueError(
                "The loaded dataset_info data must have 'data_list' and 'metainfo' keys"
            )
        metainfo = data["metainfo"]
        raw_data_list = data["data_list"]
        if isinstance(metainfo, dict):
            self.metainfo = metainfo
        else:
            raise TypeError(
                f"The metainfo loaded from dataset_info file "
                f"should be a dict, but got {type(metainfo)}!"
            )
        if isinstance(raw_data_list, list):
            self.data_list = raw_data_list
        else:
            raise TypeError(
                f"The data_list loaded from dataset_info file "
                f"should be a list, but got {type(raw_data_list)}!"
            )

    def load_info(self):
        raise NotImplementedError

    def _is_filtered(self, data, filter_cfg):
        """判断数据项是否被过滤."""
        if isinstance(filter_cfg, dict):
            for type, rule in filter_cfg.items():
                if type == "replace":
                    for key, value in rule.items():
                        # value = ("origin", "object") | {"cat": "dog"} | [("cat", "dog"), ("dog", "cat")] | [{"cat": "dog"}, {"dog": "cat"}]
                        value = [value] if not isinstance(value, list) else value
                        for item in value:
                            # item = ("origin", "object") | {"cat": "dog"}
                            if isinstance(item, tuple):
                                if key in data and data[key] == item[0]:
                                    data[key] = item[1]
                            elif isinstance(item, dict):
                                if key in data and data[key] in item:
                                    data[key] = item[data[key]]
                elif type == "drop":
                    if isinstance(rule, str):
                        rule = [rule]
                    if isinstance(rule, list):
                        for key in rule:
                            if key in data:
                                del data[key]
                        # 结束, 继续下一个过滤规则
                        continue

                    if isinstance(rule, dict):
                        for key, value in rule.items():
                            value = [value] if not isinstance(value, list) else value
                            if key in data and data[key] in value:
                                return True
                elif type == "contain":
                    for key, value in rule.items():
                        value = [value] if not isinstance(value, list) else value
                        if key in data and any(v in data[key] for v in value):
                            return True
                elif type == "query":
                    # r"(\w+)\s*([=!<>]+|in|not in)\s*([^&\s]+|\w+)"
                    matches = re.findall(r"(\w+)\s*([=!<>]+)\s*([^&\|]+)", rule)
                    if not matches:
                        raise ValueError("Invalid query format")
                    for key, operator, value in matches:
                        if key not in data:
                            return False
                        query = f"{data[key]!r}{operator}{value.strip()}"
                        if eval(query):
                            return True
        return False

    def filter(self) -> List[dict]:
        """通过指定的过滤规则过滤数据集, 默认返回所有数据."""
        filtered_data_list = []
        for data in self.data_list:
            if not self._is_filtered(data, self.filter_cfg):
                filtered_data_list.append(data)
        self.data_list = filtered_data_list
        return self.data_list

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        data_bytes = np.concatenate(data_list)
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address

    def get_data_info(self, idx: int) -> dict:
        """获取数据集中指定索引的数据信息(含 metainfo 和 datainfo)."""
        if self.serialize:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = copy.deepcopy(self.data_list[idx])

        if idx >= 0:
            data_info["index"] = idx
        else:
            data_info["index"] = len(self) + idx

        return data_info

    def save(self, name: str) -> Union[dict, List[dict]]:
        """save dataset info to local file."""
        if isfile(self.root):
            file_path = osp.join(osp.dirname(self.root), name, ".pkl")
        else:
            file_path = osp.join(self.root, name, ".pkl")

        dump(
            {"data_list": self.data_list, "metainfo": self.metainfo},
            file_path,
            file_format="pkl",
        )

    def __getitem__(self, idx: int) -> dict:
        """获取数据集中指定索引的经过处理后的数据."""
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    def get_subset(
        self, indices: Union[Sequence[int], Sequence[float], int]
    ) -> "BaseDataset":
        """获取原始数据集的子集.
        - 如果"indices"类型为"int", 则根据"indices"的正值或负值提取前几个或最后几个数据项作为子集.
        - 如果"indices"是"int"元素的序列, 则它通过选择"indices"中具有给定索引的数据项来提取子集.
        - 如果"indices"是"float"元素的序列, 并且它们的总和为 1, 则它根据指定的比率对子集进行分区, 分区的数量由元素的数量决定.
        """

        sub_datasets = []
        if isinstance(indices, int):
            assert len(self) > indices >= -len(self)
            indices_list = [
                range(indices)
                if indices >= 0
                else range(len(self) + indices, len(self))
            ]
        elif isinstance(indices, Sequence):
            if not indices:
                indices_list = [indices]
            else:
                if isinstance(indices[0], int):
                    indices_list = [indices]
                elif isinstance(indices[0], float):
                    assert np.abs(sum(indices) - 1) < 1e-5
                    indices_list = np.cumsum(indices)
                    indices_list = (indices_list * len(self)).astype(np.int32)
                    indices_list = np.insert(indices_list, 0, 0)
                    indices_list[-1] = len(self)
                    indices_list = [
                        range(indices_list[i], indices_list[i + 1])
                        for i in range(len(indices_list) - 1)
                    ]
                else:
                    raise TypeError(
                        "indices should be a int or sequence of int, but got {type(indices)}"
                    )
        else:
            raise TypeError(
                "indices should be a int or sequence of int, but got {type(indices)}"
            )
        for indices in indices_list:
            memo = dict()
            cls = self.__class__
            sub_dataset = cls.__new__(cls)
            memo[id(self)] = sub_dataset
            for key, value in self.__dict__.items():
                if key in ["data_list", "data_address", "data_bytes"]:
                    continue
                super(BaseDataset, sub_dataset).__setattr__(
                    key, copy.deepcopy(value, memo)
                )

            if self.serialize:
                sub_data_bytes = []
                sub_data_address = []
                for idx in indices:
                    assert len(self) > idx >= -len(self)
                    start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
                    end_addr = self.data_address[idx].item()
                    sub_data_bytes.append(self.data_bytes[start_addr:end_addr])
                    sub_data_address.append(end_addr - start_addr)
                if sub_data_bytes:
                    sub_data_bytes = np.concatenate(sub_data_bytes)
                    sub_data_address = np.cumsum(sub_data_address)
                else:
                    sub_data_bytes = np.array([])
                    sub_data_address = np.array([])

                sub_dataset.data_bytes = sub_data_bytes.copy()
                sub_dataset.data_address = sub_data_address.copy()
            else:
                sub_data_list = []
                for idx in indices:
                    sub_data_list.append(self.data_list[idx])
                sub_dataset.data_list = copy.deepcopy(sub_data_list)
            sub_datasets.append(sub_dataset)
        return sub_datasets

    def get_subset_(self, indices: Union[Sequence[int], Sequence[float], int]) -> None:
        """The in-place version of ``get_subset`` to convert dataset to a
        subset of original dataset.
        """
        # TODO: Implements in-place fetching of subsets.
        raise NotImplementedError

    def __len__(self) -> int:
        """获取数据集中数据项的数量."""
        if self.serialize:
            return len(self.data_address)
        else:
            return len(self.data_list)
