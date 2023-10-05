import copy
from typing import Any, Iterator, Optional, Tuple, Type, Union

import numpy as np
import torch


class BaseDataElement:
    """基础数据元素, 用于提供支持 Tensor 操作和 Dict 操作的接口.

    基于 BaseDataElement 可以实现 InstanceData, PixelData 和 LabelData, 以使用不同名称表示不同类型的真实标签或预测.

    此外, BaseDataElement 将其管理的数据分为了 ``metainfo`` 和 ``data`` 两种类型, 以实现区分管理.
        - 两者均支持类似字典或属性的访问操作或修改操作以及 "in"、 "del"、 "pop(str)"、 "get(str)"操作.
        - ``metainfo``: 通常管理有关样本数据的元信息, 例如文件名、图像形状、填充形状等. 
            提供了基本函数 "metainfo_keys()"、 "metainfo_values()"、 "metainfo_items()"、 
            "set_metainfo()"等, 用于管理 metainfo 中的数据. 
        - ``data``: 管理注释或模型预测.
            提供了基本函数 "keys()"、 "values()"、 "items()"、 "set_data"等. 用于管理 data 中的数据. 
            用户还可以将所有torch.Tensor中的张量类型方法应用于 "data_fields"中的所有张量, 例如 ".cuda()"、 ".cpu()"、 ".numpy()"、 ".to()"、 "to_tensor()"、 ".detach()". 

    Args:
        metainfo (dict, optional): 包含单个图像的元信息的字典, 例如 "dict (img_shape = (512, 512, 3), scale_factor = (1, 1, 1, 1))". 默认为None. 
        kwargs (dict, optional): 包含单个图像或模型预测的注释的字典. 默认为None. 
 
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:
        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """根据输入的字典元素, 通过 set_field 设置 metainfo 中管理的数据.
        """
        assert isinstance(
            metainfo, dict
        ), f"metainfo should be a ``dict`` but got {type(metainfo)}"
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type="metainfo", dtype=None)

    def set_data(self, data: dict) -> None:
        """根据输入的字典元素, 通过 set_field 设置 data 中管理的数据.
        """
        assert isinstance(data, dict), f"data should be a `dict` but got {data}"
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data`
            # to set property method.
            setattr(self, k, v)

    def update(self, instance: "BaseDataElement") -> None:
        """The update() method updates the BaseDataElement with the elements
        from another BaseDataElement object.

        Args:
            instance (BaseDataElement): Another BaseDataElement object for
                update the current object.
        """
        assert isinstance(
            instance, BaseDataElement
        ), f"instance should be a `BaseDataElement` but got {type(instance)}"
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: Optional[dict] = None, **kwargs) -> "BaseDataElement":
        """返回一个相同类型的新的数据元素, 如果提供了 metainfo, 则覆盖原值, 否则使用原值的备份.

        Args:
            metainfo (dict, optional): A dict contains the meta information.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                model predictions.

        Returns:
            BaseDataElement: A new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element.

        Returns:
            BaseDataElement: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {
            "_" + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        """setattr is only used to set data."""
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )
        else:
            self.set_field(name=name, value=value, field_type="data", dtype=None)

    def __delattr__(self, item: str):
        """Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ("_metainfo_fields", "_data_fields"):
            raise AttributeError(
                f"{item} has been used as a " "private attribute, which is immutable."
            )
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # Dict 类型操作
    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        """Get property in data and metainfo as the same as python."""
        # Use `getattr()` rather than `self.__dict__.get()` to allow getting
        # properties.
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        """Pop property in data and metainfo as the same as python."""
        assert len(args) < 3, "``pop`` get more than 2 arguments"
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f"{args[0]} is not contained in metainfo or data")

    def __contains__(self, item: str) -> bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(
        self,
        name: str,
        value: Any,
        dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
        field_type: str = "data",
    ) -> None:
        """为 field_type 指定的管理区域添加要管理数据.
        """
        assert field_type in ["metainfo", "data"]
        if dtype is not None:
            assert isinstance(
                value, dtype
            ), f"{value} should be a {dtype} but got {type(value)}"

        if field_type == "metainfo":
            if name in self._data_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of metainfo "
                    f"because {name} is already a data field"
                )
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of data "
                    f"because {name} is already a metainfo field"
                )
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> "BaseDataElement":
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cpu(self) -> "BaseDataElement":
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cuda(self) -> "BaseDataElement":
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def npu(self) -> "BaseDataElement":
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def mlu(self) -> "BaseDataElement":
        """Convert all tensors to MLU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.mlu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self) -> "BaseDataElement":
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def numpy(self) -> "BaseDataElement":
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> "BaseDataElement":
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        """Convert BaseDataElement to dict."""
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.all_items()
        }

    def __repr__(self) -> str:
        """Represent the object."""

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def dump(obj: Any) -> str:
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f"\n{k}: {_addindent(dump(v), 4)}"
            elif isinstance(obj, BaseDataElement):
                _repr += "\n\n    META INFORMATION"
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += "\n\n    DATA FIELDS"
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f"<{classname}({_repr}\n) at {hex(id(obj))}>"
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)
