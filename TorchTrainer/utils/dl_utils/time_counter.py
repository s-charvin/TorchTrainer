import time
from typing import Optional, Union

import torch

from TorchTrainer.utils.dist import master_only
from TorchTrainer.utils.logging import GLogger, print_log


class TimeCounter:
    """一个用于计算函数或方法平均运行时间的工具, 用户可以将其作为装饰器或上下文管理器使用, 以计算代码块的平均运行时间
    Args:
        log_interval (int): The interval of logging. Defaults to 1.
        warmup_interval (int): The interval of warmup. Defaults to 1.
        with_sync (bool): Whether to synchronize cuda. Defaults to True.
        tag (str, optional): Function tag. Used to distinguish between
            different functions or methods being called. Defaults to None.
        logger (GLogger, optional): Formatted logger used to record messages.
                Defaults to None.
    """

    instance_dict: dict = dict()

    log_interval: int
    warmup_interval: int
    logger: Optional[GLogger]
    __count: int
    __pure_inf_time: float

    def __new__(
        cls,
        log_interval: int = 1,
        warmup_interval: int = 1,
        with_sync: bool = True,
        tag: Optional[str] = None,
        logger: Optional[GLogger] = None,
    ):
        assert warmup_interval >= 1
        if tag is not None and tag in cls.instance_dict:
            return cls.instance_dict[tag]

        instance = super().__new__(cls)
        cls.instance_dict[tag] = instance

        instance.log_interval = log_interval
        instance.warmup_interval = warmup_interval
        instance.with_sync = with_sync
        instance.tag = tag
        instance.logger = logger

        instance.__count = 0
        instance.__pure_inf_time = 0.0
        instance.__start_time = 0.0

        return instance

    @master_only
    def __call__(self, fn):
        if self.tag is None:
            self.tag = fn.__name__

        def wrapper(*args, **kwargs):
            self.__count += 1

            if self.with_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            result = fn(*args, **kwargs)

            if self.with_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            self.print_time(elapsed)

            return result

        return wrapper

    @master_only
    def __enter__(self):
        assert self.tag is not None, (
            "In order to clearly distinguish "
            "printing information in different "
            "contexts, please specify the "
            "tag parameter"
        )

        self.__count += 1

        if self.with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.__start_time = time.perf_counter()

    @master_only
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.__start_time
        self.print_time(elapsed)

    def print_time(self, elapsed: Union[int, float]) -> None:
        """print times per count."""
        if self.__count >= self.warmup_interval:
            self.__pure_inf_time += elapsed

            if self.__count % self.log_interval == 0:
                times_per_count = (
                    1000
                    * self.__pure_inf_time
                    / (self.__count - self.warmup_interval + 1)
                )
                print_log(
                    f"[{self.tag}]-time per run averaged in the past "
                    f"{self.__count} runs: {times_per_count:.1f} ms",
                    self.logger,
                )
