import logging
import os
import os.path as osp
import sys
import warnings
from getpass import getuser
from logging import Logger, LogRecord
from socket import gethostname
from typing import Optional, Union

from termcolor import colored

from TorchTrainer.utils import GlobalManager
from TorchTrainer.utils.manager import _accquire_lock, _release_lock


class FilterDuplicateWarning(logging.Filter):
    """过滤重复的警告信息."""

    def __init__(self, name: str = "TorchTrainer"):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        """过滤重复的警告信息.
        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


class Formatter(logging.Formatter):
    """GLogger的格式化类.
    如果日志级别为 error, logger 还会额外输出代码的位置
    Args:
        color (bool): Whether to use colorful format. filehandler is not
            allowed to use color format, otherwise it will be garbled.
        blink (bool): Whether to blink the ``INFO`` and ``DEBUG`` logging
            level.
        **kwargs: Keyword arguments passed to
            :meth:`logging.Formatter.__init__`.
    """

    _color_mapping: dict = dict(
        ERROR="red", WARNING="yellow", INFO="white", DEBUG="green"
    )

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (
            not color and blink
        ), "blink should only be available when color is True"
        # Get prefix format according to color.
        error_prefix = self._get_prefix("ERROR", color, blink=True)
        warn_prefix = self._get_prefix("WARNING", color, blink=True)
        info_prefix = self._get_prefix("INFO", color, blink)
        debug_prefix = self._get_prefix("DEBUG", color, blink)

        # Config output format.
        self.err_format = (
            f"%(asctime)s - %(name)s - {error_prefix} - "
            "%(pathname)s - %(funcName)s - %(lineno)d - "
            "%(message)s"
        )
        self.warn_format = f"%(asctime)s - %(name)s - {warn_prefix} - %(" "message)s"
        self.info_format = f"%(asctime)s - %(name)s - {info_prefix} - %(" "message)s"
        self.debug_format = f"%(asctime)s - %(name)s - {debug_prefix} - %(" "message)s"

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        """获得目标日志级别的前缀.

        Args:
            level (str): log level.
            color (bool): Whether to get colorful prefix.
            blink (bool): Whether the prefix will blink.

        Returns:
            str: The plain or colorful prefix.
        """
        if color:
            attrs = ["underline"]
            if blink:
                attrs.append("blink")
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        """重写 ``logging.Formatter.format`` 方法. 根据日志级别输出信息.

        Args:
            record (LogRecord): A LogRecord instance represents an event being
                logged.

        Returns:
            str: Formatted result.
        """
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class GLogger(Logger, GlobalManager):
    """用于记录日志的格式化类.

    ``GLogger`` 可以创建格式化的日志记录器, 以不同的日志级别记录日志消息, 并以与 ``GlobalManager`` 相同的方式获取全局实例.
    ``GLogger`` 具有以下功能:
        - 分布式日志存储, ``GLogger`` 可以根据 `log_file` 选择是否保存不同排名的日志.
        - 在终端上显示时, 具有不同颜色和格式的不同日志级别的消息.

    注意:
        - 日志记录器的 `name` 和 ``GLogger`` 的 ``instance_name`` 可能不同.
          只能通过 ``GLogger.get_instance`` 而不是 ``logging.getLogger``来获取 ``GLogger`` 实例.
          此功能确保 ``GLogger`` 不会受到第三方日志配置的影响.
        - 与 ``logging.Logger`` 不同, ``GLogger`` 不会记录没有 ``Handler``
            的警告或错误消息.

    Args:
        name (str): Global instance name.
        logger_name (str): ``name`` attribute of ``Logging.Logger`` instance.
            If `logger_name` is not defined, defaults to 'TorchTrainer'.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level (str): The log level of the handler. Defaults to
            'INFO'. If log level is 'DEBUG', distributed logs will be saved
            during distributed training.
        file_mode (str): The file mode used to open log file. Defaults to 'w'.
        distributed (bool): Whether to save distributed logs, Defaults to
            false.
    """

    def __init__(
        self,
        name: str,
        logger_name="TorchTrainer",
        log_file: Optional[str] = None,
        log_level: Union[int, str] = "INFO",
        file_mode: str = "w",
        distributed=False,
    ):
        Logger.__init__(self, logger_name)
        GlobalManager.__init__(self, name)
        # Get rank in DDP mode.
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]
        global_rank = _get_rank()
        device_id = _get_device_id()

        # Config stream_handler. If `rank != 0`. stream_handler can only
        # export ERROR logs.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # `StreamHandler` record month, day, hour, minute, and second
        # timestamp.
        stream_handler.setFormatter(Formatter(color=True, datefmt="%m/%d %H:%M:%S"))
        # Only rank0 `StreamHandler` will log messages below error level.
        if global_rank == 0:
            stream_handler.setLevel(log_level)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)

        if log_file is not None:
            world_size = _get_world_size()
            is_distributed = (
                log_level <= logging.DEBUG or distributed
            ) and world_size > 1
            if is_distributed:
                filename, suffix = osp.splitext(osp.basename(log_file))
                hostname = _get_host_info()
                if hostname:
                    filename = (
                        f"{filename}_{hostname}_device{device_id}_"
                        f"rank{global_rank}{suffix}"
                    )
                else:
                    # Omit hostname if it is empty
                    filename = (
                        f"{filename}_device{device_id}_" f"rank{global_rank}{suffix}"
                    )
                log_file = osp.join(osp.dirname(log_file), filename)
            # Save multi-ranks logs if distributed is True. The logs of rank0
            # will always be saved.
            if global_rank == 0 or is_distributed:
                # Here, the default behaviour of the official logger is 'a'.
                # Thus, we provide an interface to change the file mode to
                # the default behaviour. `FileHandler` is not supported to
                # have colors, otherwise it will appear garbled.
                file_handler = logging.FileHandler(log_file, file_mode)
                # `StreamHandler` record year, month, day hour, minute,
                # and second timestamp. file_handler will only record logs
                # without color to avoid garbled code saved in files.
                file_handler.setFormatter(
                    Formatter(color=False, datefmt="%Y/%m/%d %H:%M:%S")
                )
                file_handler.setLevel(log_level)
                file_handler.addFilter(FilterDuplicateWarning(logger_name))
                self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_current_instance(cls) -> "GLogger":
        """获取最新创建的 ``GLogger`` 实例. ``GLogger`` 可以在创建任何实例之前调用 ``get_current_instance`` 并返回一个实例名为 "TorchTrainer" 的日志记录器.
        Returns:
            GLogger: Configured logger instance.
        """
        if not cls._instance_dict:
            cls.get_instance("TorchTrainer")
        return super().get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        """将日志记录传递给所有相关的处理程序.
        在 ``logging.Logger`` 中重写 ``callHandlers`` 方法, 以避免在 DDP 模式下出现多个警告消息.
        循环遍历日志记录器实例及其父级在日志记录器层次结构中的所有处理程序. 如果未找到处理程序, 则不会输出记录.

        Args:
            record (LogRecord): A ``LogRecord`` instance contains logged
                message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        """设置日志记录器的日志级别.
        如果调用 ``logging.Logger.selLevel``, ``logging.Manager`` 管理的所有 ``logging.Logger`` 实例都将清除缓存.
        由于 ``GLogger`` 不再由 ``logging.Manager`` 管理, 因此 ``GLogger`` 应重写此方法, 以清除 ``GlobalManager`` 管理的所有 ``GLogger`` 实例的缓存.

        level must be an int or a str.
        """
        self.level = logging._checkLevel(level)
        _accquire_lock()
        # The same logic as `logging.Manager._clear_cache`.
        for logger in GLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()


def print_log(
    msg, logger: Optional[Union[Logger, str]] = None, level=logging.INFO
) -> None:
    """用于在终端上打印日志消息的函数.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif logger == "current":
        logger_instance = GLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        if GLogger.check_instance_created(logger):
            logger_instance = GLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f"GLogger: {logger} has not been created!")
    else:
        raise TypeError(
            "`logger` should be either a logging.Logger object, str, "
            f'"silent", "current" or None, but got {type(logger)}'
        )


def _get_world_size():
    """获得当前机器的总排名. 如果没有 torch, 则返回 1."""
    try:
        # requires torch
        from TorchTrainer.utils.dist import get_world_size
    except ImportError:
        return 1
    else:
        return get_world_size()


def _get_rank():
    """获得当前机器的排名. 如果没有 torch, 则返回 0."""
    try:
        # requires torch
        from TorchTrainer.utils.dist import get_rank
    except ImportError:
        return 0
    else:
        return get_rank()


def _get_device_id():
    """获得当前机器的设备 id. 如果没有 torch, 则返回 0."""

    try:
        import torch
    except ImportError:
        return 0
    else:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        # TODO: return device id of npu and mlu.
        if not torch.cuda.is_available():
            return local_rank
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is None:
            num_device = torch.cuda.device_count()
            cuda_visible_devices = list(range(num_device))
        else:
            cuda_visible_devices = cuda_visible_devices.split(",")
        try:
            return int(cuda_visible_devices[local_rank])
        except ValueError:
            # handle case for Multi-Instance GPUs
            # see #1148 for details
            return cuda_visible_devices[local_rank]


def _get_host_info() -> str:
    """获得当前机器的主机名和用户名.
    """
    host = ""
    try:
        host = f"{getuser()}@{gethostname()}"
    except Exception as e:
        warnings.warn(f"Host or user not found: {str(e)}")
    finally:
        return host
