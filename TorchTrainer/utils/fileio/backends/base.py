import logging
from abc import ABCMeta, abstractmethod

from TorchTrainer.utils.logging import print_log


class BaseStorageBackend(metaclass=ABCMeta):
    """一个抽象类, 用于定义存储后端的接口.
    所有的后端都需要实现两个接口: :meth:`get()` 和 :meth:`get_text()`.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    # This attribute will be deprecated in future.
    _allow_symlink = False

    @property
    def allow_symlink(self):
        print_log(
            "allow_symlink will be deprecated in future",
            logger="current",
            level=logging.WARNING,
        )
        return self._allow_symlink

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass
