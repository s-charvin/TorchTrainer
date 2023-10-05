import copy
import datetime
import re
from collections import OrderedDict
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch

from TorchTrainer.utils.device import get_max_cuda_memory, is_cuda_available
from TorchTrainer.utils.registry import LOG_PROCESSORS


@LOG_PROCESSORS.register_module()
class LogProcessor:
    """一个用于格式化从``runner.message_hub.log_scalars``收集的日志信息的日志处理器

    ``LogProcessor``实例由运行程序构建, 并将``runner.message_hub.log_scalars``格式化为 ``tag`` 和 ``log_str``, 可以直接被 ``LoggerHook`` 和 ``GLogger`` 使用.
    此外, 构造函数的参数 ``custom_cfg`` 可以控制日志的统计方法.


    Args:
        window_size (int): 默认平滑间隔, 默认值为10.
        by_epoch (bool): 是否使用 epoch 样式格式化日志, 默认值为 True.
        custom_cfg (list[dict], optional): 包含多个日志配置字典, 其中 key 表示日志的数据源名称, value 表示用于计算数据源的统计方法和相应参数. 默认值为 None.

            - 如果custom_cfg是None, 则所有日志都将通过默认方法进行格式化, 例如默认窗口大小平滑损失. 如果 custom_cfg 定义为配置字典列表,
              例如：[dict(data_src=loss, method='mean', log_name='global_loss', window_size='global')].
              这意味着日志项``loss``将被计为全局平均值, 并且还将作为``global_loss``记录. 如果在配置字典中未定义``log_name``, 则将覆盖原始记录的键.
            - 原始记录项不能被重写两次. 以下是一个错误示例：

              [dict(data_src=loss, method='mean', window_size='global'), dict(data_src=loss, method='mean', window_size='epoch')].
              custom_cfg中的两个日志配置字典都没有`log_name`键, 这意味着损失项将被重写两次.

            - For those statistic methods with the ``window_size`` argument,
              if ``by_epoch`` is set to False, ``windows_size`` should not be
              `epoch` to statistics log value by epoch.
        num_digits (int): 日志消息中显示有效数字位数
        log_with_hierarchy (bool): 是否按层级记录日志. 如果是True, 则信息会以层级方式写入可视化后端,
            例如, :obj:`LocalVisBackend` 和 :obj:`TensorboardBackend` 后端 “loss”会保存成 “train/loss”,
            默认值为False.
        mean_pattern (str): 一个正则表达式, 用于匹配需要包含在平滑统计中的日记.
    """

    def __init__(
        self,
        window_size=10,
        by_epoch=True,
        custom_cfg: Optional[List[dict]] = None,
        num_digits: int = 4,
        log_with_hierarchy: bool = False,
        mean_pattern=r".*(loss|time|data_time|grad_norm).*",
    ):
        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else []
        self.num_digits = num_digits
        self.log_with_hierarchy = log_with_hierarchy
        self.mean_pattern = re.compile(mean_pattern)
        self._check_custom_cfg()

    def get_log_after_iter(self, runner, batch_idx: int, mode: str) -> Tuple[dict, str]:
        """在训练、验证或测试周期之后, 格式化日志字符串.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner, train, test or val.

        Return:
            Tuple(dict, str): 格式化后的日志字典/字符串, 将被记录到 :obj:`runner.message_hub` 和 :obj:`runner.visualizer` 中.
        """
        assert mode in ["train", "test", "val"]
        # 将 ``custom_cfg`` 中定义的 ``window_size`` 覆盖为整数值.
        parsed_cfg = self._parse_windows_size(runner, batch_idx, self.custom_cfg)
        # og_tag 用于将日志信息写入终端
        # 如果 `self.log_with_hierarchy` 为 False, 则标签与 log_tag 相同.
        # 否则, tag 中的每个键都以前缀 `train`、`test` 或 `val` 开头.

        log_tag = self._collect_scalars(parsed_cfg, runner, mode)

        if not self.log_with_hierarchy:
            tag = copy.deepcopy(log_tag)
        else:
            tag = self._collect_scalars(parsed_cfg, runner, mode, True)

        # 记录学习率
        lr_str_list = []
        for key, value in tag.items():
            if key.endswith("lr"):
                key = self._remove_prefix(key, f"{mode}/")
                log_tag.pop(key)
                lr_str_list.append(f"{key}: " f"{value:.{self.num_digits}e}")
        lr_str = " ".join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
        if self.by_epoch:
            # Align the iteration log:
            # Epoch(train)  [  9][010/270]
            # ...                 ||| |||
            # Epoch(train)  [ 10][100/270]
            dataloader_len = self._get_dataloader_size(runner, mode)
            cur_iter = self._get_iter(runner, batch_idx)
            cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))
            if mode in ["train", "val"]:
                cur_epoch = self._get_epoch(runner, mode)
                if not (
                    isinstance(runner._train_loop, dict) or runner._train_loop is None
                ):
                    # Right Align the epoch log:
                    # Epoch(train)   [9][100/270]
                    # ...             ||
                    # Epoch(train) [100][100/270]
                    max_epochs = runner.max_epochs
                    # 3 means the three characters: "[", "]", and " " occupied
                    # in " [{max_epochs}]"
                    cur_epoch_str = f"[{cur_epoch}]".rjust(
                        len(str(max_epochs)) + 3, " "
                    )
                else:
                    cur_epoch_str = f"[{cur_epoch}]"
                tag["epoch"] = cur_epoch
                log_str = (
                    f"Epoch({mode}){cur_epoch_str}"
                    f"[{cur_iter_str}/{dataloader_len}]  "
                )
            else:
                log_str = f"Epoch({mode}) " f"[{cur_iter_str}/{dataloader_len}]  "
        else:
            if mode == "train":
                cur_iter = self._get_iter(runner, batch_idx)
                cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
                log_str = f"Iter({mode}) " f"[{cur_iter_str}/{runner.max_iters}]  "
            else:
                dataloader_len = self._get_dataloader_size(runner, mode)
                cur_iter_str = str(batch_idx + 1).rjust(len(str(dataloader_len)))
                log_str = f"Iter({mode}) [{cur_iter_str}/{dataloader_len}]  "

        # Add global iter.
        if isinstance(runner._train_loop, dict) or runner._train_loop is None:
            tag["iter"] = 0
        else:
            tag["iter"] = runner.iter + 1

        # Concatenate lr, momentum string with log header.
        log_str += f"{lr_str}  "
        # 如果在运行器中使用了 IterTimerHook, 应记录 eta、time 和 data_time
        if (
            all(item in log_tag for item in ["time", "data_time"])
            and "eta" in runner.message_hub.runtime_info
        ):
            eta = runner.message_hub.get_info("eta")
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += f"eta: {eta_str}  "
            log_str += (
                f'time: {log_tag["time"]:.{self.num_digits}f}  '
                f"data_time: "
                f'{log_tag["data_time"]:.{self.num_digits}f}  '
            )
            # Pop recorded keys
            log_tag.pop("time")
            log_tag.pop("data_time")

        # 如果cuda可用, 则应计算占用的最大内存
        if is_cuda_available():
            max_memory = self._get_max_memory(runner)
            log_str += f"memory: {max_memory}  "
            tag["memory"] = max_memory
        # Loop left keys to fill `log_str`.
        if mode in ("train", "val"):
            log_items = []
            for name, val in log_tag.items():
                if mode == "val" and not name.startswith("val/loss"):
                    continue
                if isinstance(val, float):
                    val = f"{val:.{self.num_digits}f}"
                log_items.append(f"{name}: {val}")
            log_str += "  ".join(log_items)
        return tag, log_str

    def get_log_after_epoch(
        self, runner, batch_idx: int, mode: str, with_non_scalar: bool = False
    ) -> Tuple[dict, str]:
        """在验证或测试轮次之后, 格式化日志字符串

        Args:
            runner (Runner): The runner of validation/testing phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner.
            with_non_scalar (bool): Whether to include non-scalar infos in the
                returned tag. Defaults to False.

        Return:
            Tuple(dict, str): Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in ["test", "val"], (
            "`_get_metric_log_str` only accept val or test mode, but got " f"{mode}"
        )
        dataloader_len = self._get_dataloader_size(runner, mode)

        # By epoch:
        #     Epoch(val) [10][1000/1000]  ...
        #     Epoch(test) [1000/1000] ...
        # By iteration:
        #     Iteration(val) [1000/1000]  ...
        #     Iteration(test) [1000/1000]  ...
        if self.by_epoch:
            if mode == "val":
                cur_epoch = self._get_epoch(runner, mode)
                log_str = (
                    f"Epoch({mode}) [{cur_epoch}][{dataloader_len}/"
                    f"{dataloader_len}]  "
                )
            else:
                log_str = f"Epoch({mode}) [{dataloader_len}/{dataloader_len}]  "

        else:
            log_str = f"Iter({mode}) [{dataloader_len}/{dataloader_len}]  "

        custom_cfg_copy = copy.deepcopy(self.custom_cfg)
        # remove prefix
        custom_keys = [
            self._remove_prefix(cfg["data_src"], f"{mode}/") for cfg in custom_cfg_copy
        ]
        # Count the averaged time and data_time by epoch
        if "time" not in custom_keys:
            custom_cfg_copy.append(
                dict(data_src="time", window_size="epoch", method_name="mean")
            )
        if "data_time" not in custom_keys:
            custom_cfg_copy.append(
                dict(data_src="data_time", window_size="epoch", method_name="mean")
            )
        parsed_cfg = self._parse_windows_size(runner, batch_idx, custom_cfg_copy)
        # tag is used to write log information to different backends.
        ori_tag = self._collect_scalars(
            parsed_cfg, runner, mode, self.log_with_hierarchy
        )
        non_scalar_tag = self._collect_non_scalars(runner, mode)
        # move `time` or `data_time` to the end of the log
        tag = OrderedDict()
        time_tag = OrderedDict()
        for key, value in ori_tag.items():
            if key in (f"{mode}/time", f"{mode}/data_time", "time", "data_time"):
                time_tag[key] = value
            else:
                tag[key] = value
        # Log other messages.
        log_items = []
        log_str += "  "
        for name, val in chain(tag.items(), non_scalar_tag.items(), time_tag.items()):
            if isinstance(val, float):
                val = f"{val:.{self.num_digits}f}"
            if isinstance(val, (torch.Tensor, np.ndarray)):
                # newline to display tensor and array.
                val = f"\n{val}\n"
            log_items.append(f"{name}: {val}")
        log_str += "  ".join(log_items)

        if with_non_scalar:
            tag.update(non_scalar_tag)
        tag.update(time_tag)
        return tag, log_str

    def _collect_scalars(
        self, custom_cfg: List[dict], runner, mode: str, reserve_prefix: bool = False
    ) -> dict:
        """根据模式收集日志信息, 组成一个字典.

        Args:
            custom_cfg (List[dict]): A copy of ``self.custom_cfg`` with int
                ``window_size``.
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.
            reserve_prefix (bool): Whether to reserve the prefix of the key.

        Returns:
            dict: Statistical values of logs.
        """
        custom_cfg = copy.deepcopy(custom_cfg)
        tag = OrderedDict()

        history_scalars = runner.message_hub.log_scalars  # 训练/验证/测试阶段的历史标量
        mode_history_scalars = OrderedDict()  # 对应模式的历史标量

        # 提取日志标量并根据模式删除前缀到`mode_history_scalars`中
        for prefix_key, log_buffer in history_scalars.items():
            if prefix_key.startswith(mode):
                if not reserve_prefix:
                    key = self._remove_prefix(prefix_key, f"{mode}/")
                else:
                    key = prefix_key
                mode_history_scalars[key] = log_buffer
        for key in mode_history_scalars:
            # 更新最新的学习率和平滑时间日志
            if re.search(self.mean_pattern, key) is not None:
                tag[key] = mode_history_scalars[key].mean(self.window_size)
            else:
                # 默认统计方法为当前值
                tag[key] = mode_history_scalars[key].current()
        # 更新自定义键
        for log_cfg in custom_cfg:
            if not reserve_prefix:
                data_src = log_cfg.pop("data_src")
                log_name = f"{log_cfg.pop('log_name', data_src)}"
            else:
                data_src = f"{mode}/{log_cfg.pop('data_src')}"
                log_name = f"{mode}/{log_cfg.pop('log_name', data_src)}"
            # 自定义配置中的日志项只能存在于训练或验证模式中
            if data_src in mode_history_scalars:
                tag[log_name] = mode_history_scalars[data_src].statistics(**log_cfg)
        return tag

    def _collect_non_scalars(self, runner, mode: str) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            dict: non-scalar infos of the specified mode.
        """
        # infos of train/val/test phase.
        infos = runner.message_hub.runtime_info
        # corresponding mode infos
        mode_infos = OrderedDict()
        # extract log info and remove prefix to `mode_infos` according to mode.
        for prefix_key, value in infos.items():
            if prefix_key.startswith(mode):
                if self.log_with_hierarchy:
                    key = prefix_key
                else:
                    key = self._remove_prefix(prefix_key, f"{mode}/")
                mode_infos[key] = value
        return mode_infos

    def _remove_prefix(self, string: str, prefix: str):
        """Remove the prefix ``train``, ``val`` and ``test`` of the key."""
        if string.startswith(prefix):
            return string[len(prefix) :]
        else:
            return string

    def _check_custom_cfg(self) -> None:
        """Check the legality of ``self.custom_cfg``."""

        def _check_window_size():
            for log_cfg in self.custom_cfg:
                if not self.by_epoch:
                    assert log_cfg["window_size"] != "epoch", (
                        "window_size cannot be epoch if LoggerHook.by_epoch"
                        " is False."
                    )

        def _check_repeated_log_name():
            # The `log_name` of the same data_src should not be repeated.
            # If `log_name` is not specified, `data_src` will be overwritten.
            # But only allowed to be overwritten once.
            check_set = set()
            for log_cfg in self.custom_cfg:
                assert "data_src" in log_cfg
                data_src = log_cfg["data_src"]
                log_name = log_cfg.get("log_name", data_src)
                assert log_name not in check_set, (
                    f"Found duplicate {log_name} for {data_src}. Please check"
                    "your `custom_cfg` for `log_processor`. You should "
                    f"neither define duplicate `{log_name}` for {data_src} "
                    f"nor do not define any {log_name} for multiple "
                    f"{data_src}, See more information in the docstring of "
                    "LogProcessor"
                )

                check_set.add(log_name)

        _check_repeated_log_name()
        _check_window_size()

    def _parse_windows_size(
        self, runner, batch_idx: int, custom_cfg: Optional[list] = None
    ) -> list:
        """将 ``custom_cfg`` 中定义的 ``window_size`` 覆盖为整数值.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current dataloader.
            custom_cfg (list): A copy of ``self.custom_cfg``. Defaults to None
                to keep backward compatibility.
        """
        if custom_cfg is None:
            custom_cfg = copy.deepcopy(self.custom_cfg)
        else:
            custom_cfg = copy.deepcopy(custom_cfg)
        for log_cfg in custom_cfg:
            window_size = log_cfg.get("window_size", None)
            if window_size is None or isinstance(window_size, int):
                continue
            elif window_size == "epoch":
                log_cfg["window_size"] = batch_idx + 1
            elif window_size == "global":
                log_cfg["window_size"] = runner.iter + 1
            else:
                raise TypeError(
                    "window_size should be int, epoch or global, but got "
                    f"invalid {window_size}"
                )
        return custom_cfg

    def _get_max_memory(self, runner) -> int:
        """Returns the maximum GPU memory occupied by tensors in megabytes (MB)
        for a given device.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.

        Returns:
            The maximum GPU memory occupied by tensors in megabytes for a given
            device.
        """

        device = getattr(runner.model, "output_device", None)
        return get_max_cuda_memory(device)

    def _get_iter(self, runner, batch_idx: int) -> int:
        """Get current iteration index.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current
                dataloader. Defaults to None.

        Returns:
            int: The current global iter or inner iter.
        """
        if self.by_epoch:
            current_iter = batch_idx + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def _get_epoch(self, runner, mode: str) -> int:
        """Get current epoch according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            int: The current epoch.
        """
        if mode == "train":
            epoch = runner.epoch + 1
        elif mode == "val":
            if isinstance(runner._train_loop, dict) or runner._train_loop is None:
                epoch = 0
            else:
                # normal val mode
                # runner.epoch += 1 has been done before validation
                epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', but got {mode}")
        return epoch

    def _get_cur_loop(self, runner, mode: str):
        """Get current loop according to mode.

        Args:
            runner (Runner): The runner of the training/validation/testing
                process.
            mode (str): Current mode of runner.

        Returns:
            BaseLoop: Current loop of runner.
        """
        # returns type hint will occur circular import
        if mode == "train":
            return runner.train_loop
        elif mode == "val":
            return runner.val_loop
        else:
            return runner.test_loop

    def _get_dataloader_size(self, runner, mode) -> int:
        """Get dataloader size of current loop.

        Args:
            runner (Runner): The runner of the training/validation/testing
            mode (str): Current mode of runner.

        Returns:
            int: The dataloader size of current loop.
        """
        return len(self._get_cur_loop(runner=runner, mode=mode).dataloader)
