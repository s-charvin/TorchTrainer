import inspect
import threading
import warnings
from collections import OrderedDict
from typing import Type, TypeVar

_lock = threading.RLock()
T = TypeVar("T")


def _accquire_lock() -> None:
    """获取锁, 以序列化对共享数据的访问. 在使用完毕后, 请通过_release_lock()释放该锁."""
    if _lock:
        _lock.acquire()


def _release_lock() -> None:
    """释放通过调用_acquire_lock()获取的模块级锁."""
    if _lock:
        _lock.release()


class ManagerMeta(type):
    """通过元类, 使得全局类在创建实例之前先创建 _instance_dict 成员;"""

    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert "name" in params_names, f"{cls} must have the `name` argument"
        super().__init__(*args)


class GlobalManager(metaclass=ManagerMeta):
    """``GlobalManager`` 具有全局访问要求的类的基类.
    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: str = "", **kwargs):
        assert (
            isinstance(name, str) and name
        ), "name argument must be an non-empty string."
        self._instance_name = name

    @classmethod
    def get_instance(cls: Type[T], name: str, **kwargs) -> T:
        """如果名称存在, 则按名称获取子类实例, 否则创建一个新实例."""
        _accquire_lock()
        assert isinstance(name, str), f"type of name should be str, but got {type(cls)}"
        instance_dict = cls._instance_dict
        # Get the instance by name.
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)
            instance_dict[name] = instance
        elif kwargs:
            warnings.warn(
                f"{cls} instance named of {name} has been created, "
                "the method `get_instance` should not accept any other "
                "arguments"
            )
        # Get latest instantiated instance or root instance.
        _release_lock()
        return instance_dict[name]

    @classmethod
    def get_current_instance(cls):
        """获取最新创建的实例."""
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(
                f"Before calling {cls.__name__}.get_current_instance(), you "
                "should call get_instance(name=xxx) at least once."
            )
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        """检查名称对应的实例是否存在."""
        return name in cls._instance_dict

    @property
    def instance_name(self) -> str:
        """获取实例名称."""
        return self._instance_name
