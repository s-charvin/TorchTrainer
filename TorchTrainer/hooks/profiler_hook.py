import logging
import os
import os.path as osp
import sys
from typing import Callable, Optional, Union

import torch

from TorchTrainer.utils.dist import master_only
from TorchTrainer.hooks import Hook
from TorchTrainer.utils.logging import print_log
from TorchTrainer.utils.registry import HOOKS


def check_kineto() -> bool:  # noqa
    kineto_exist = False
    try:
        if torch.autograd.kineto_available():
            kineto_exist = True
    except AttributeError:
        print_log("NO KINETO", logger="current", level=logging.WARNING)
    return kineto_exist


@HOOKS.register_module()
class ProfilerHook(Hook):
    """用于在训练和推理过程中分析性能的钩子. 

    PyTorch Profiler 是一种工具, 允许在训练过程中收集性能指标(例如, CPU/GPU时间, 内存分配/释放, FLOPS 等) 并将其可视化.
    参数:
        by_epoch (bool): 按epoch或迭代进行性能分析. 默认为True. 
        profile_times (int): 由 Profiler 记录的周期 (epoch/iter) 的数量. 默认为1. 例如, profile_iters=10且by_epoch=False, 表示记录0-10个迭代. 
        activity_with_cpu (bool): 在分析中使用的活动 (CPU) 
        activity_with_cuda (bool): 在分析中使用的活动 (CUDA) 
        schedule (dict, optional): 传递给`torch.profile.schedule <https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule>`_的关键字参数. 默认为None, 表示无计划的性能分析. 
        on_trace_ready (callable, dict, optional): 处理程序或生成处理程序的字典. 默认为None, 表示没有on_trace_ready的性能分析. Callable类型需要构建自己的函数来处理'torch.autograd.profiler.profile'. 提供了两种官方推荐的方式：

            - ``schedule=dict(type='log_trace')``：在终端打印性能分析结果. 详细信息请参阅`PyTorch官方教程`_. 可配置的参数与``prof.key_averages().table``相同. 
            - ``scheduler=dict(type='tb_trace')``：使用tensorboard进行性能分析. 详细信息请参阅教程`使用tensorboard进行性能分析`_. 

        record_shapes (bool): 保存运算符输入形状的信息. 默认为False. 
        profile_memory (bool): 跟踪张量的内存分配/释放. 默认为False. 
        with_stack (bool): 记录操作的源信息 (文件和行号) . 默认为False. 
        with_flops (bool): 使用公式估计特定运算符 (矩阵乘法和2D卷积) 的FLOPS. 默认为False. 
        json_trace_path (str, optional): 以Chrome JSON格式导出收集的跟踪信息. Chrome使用'chrome://tracing'查看json文件. 默认为None, 表示性能分析不存储json文件. 

    警告:
        在达到``profile_times``之前, Profiler将在``profile_times``个迭代后自动关闭. 请确保您的调度器配置不会在迭代达到``profile_times``之前关闭Profiler. 

    .. _PyTorch官方教程: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-execution-time
    .. _使用tensorboard进行性能分析: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#pytorch-profiler-with-tensorboard
    """

    priority = "VERY_LOW"

    def __init__(
        self,
        *,
        by_epoch: bool = True,
        profile_times: int = 1,
        activity_with_cpu: bool = True,
        activity_with_cuda: bool = False,
        schedule: Optional[dict] = None,
        on_trace_ready: Union[Callable, dict, None] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        json_trace_path: Optional[str] = None,
    ) -> None:
        try:
            from torch import profiler
        except ImportError:
            raise ImportError("please upgrade torch above 1.8.1")
        if not check_kineto():
            raise ImportError(
                "Due to Kineto support issues, please upgrade "
                "pytorch above 1.8.1(windows users above 1.9.1)"
            )

        assert isinstance(by_epoch, bool), "``by_epoch`` should be a boolean."
        self.by_epoch = by_epoch

        if profile_times < 1:
            raise ValueError(
                "profile_iters should be greater than 0, " f"but got {profile_times}"
            )
        if by_epoch and profile_times > 1:
            raise ValueError(
                f"Profiler will profile 0-{profile_times} epochs.\n"
                "Since profiler will slow down the training, it is recommended"
                " to train 1 epoch with ProfilerHook and adjust your setting "
                "according to the profiler summary.\n"
                "During normal training(epoch > 1), "
                "you may disable the ProfilerHook."
            )
        self.profile_times = profile_times

        assert isinstance(
            activity_with_cpu, bool
        ), "``activity_with_cpu`` should be a boolean."
        assert isinstance(
            activity_with_cuda, bool
        ), "``activity_with_cuda`` should be a boolean."
        self.activities = []
        if activity_with_cpu:
            self.activities.append(profiler.ProfilerActivity.CPU)
        if activity_with_cuda:
            self.activities.append(profiler.ProfilerActivity.CUDA)

        if schedule is not None:
            assert isinstance(schedule, dict), "``schedule`` should be a dict."
            self.schedule = profiler.schedule(**schedule)
        else:
            self.schedule = None

        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops

        self.json_trace_path = json_trace_path
        self._closed = False

    def before_run(self, runner):
        """Initialize the profiler.

        Through the runner parameter, the validity of the parameter is further
        determined.
        """
        max_times = runner.max_epochs if self.by_epoch else runner.max_iters
        if max_times < self.profile_times:
            raise ValueError(
                f"``profile_times`` should not be greater than {max_times}"
            )

        on_trace_ready = self._parse_trace_config(runner)

        self.profiler = torch.profiler.profile(  # noqa
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )

        self.profiler.__enter__()
        runner.logger.info("profiler is profiling...")

    def _parse_trace_config(self, runner):
        """Used to parse the parameter 'on_trace_ready'."""
        if self.on_trace_ready is None:
            _on_trace_ready = None
        elif callable(self.on_trace_ready):
            _on_trace_ready = self.on_trace_ready
        elif isinstance(self.on_trace_ready, dict):
            trace_cfg = self.on_trace_ready.copy()
            trace_type = trace_cfg.pop("type")

            # Build a log printing handle
            if trace_type == "log_trace":

                def _log_handler(_profile):
                    print(_profile.key_averages().table(**trace_cfg))

                _on_trace_ready = _log_handler

            elif trace_type == "tb_trace":  # tensorboard_trace handler
                try:
                    import torch_tb_profiler
                except ImportError:
                    raise ImportError("please run ``pip install torch-tb-profiler``")

                if "dir_name" not in trace_cfg:
                    trace_cfg["dir_name"] = osp.join(runner.log_dir, "tf_tracing_logs")
                elif not osp.isabs(trace_cfg["dir_name"]):
                    trace_cfg["dir_name"] = osp.join(
                        runner.log_dir, trace_cfg["dir_name"]
                    )
                runner.logger.info(
                    "trace_files of ProfilerHook will be "
                    f'saved to {trace_cfg["dir_name"]}.'
                )

                if self.json_trace_path is not None:
                    runner.logger.warn(
                        "When using tensorboard_trace, it is recommended to "
                        "save json files by setting ``worker_name`` instead of"
                        " setting ``json_trace_path``"
                    )
                _on_trace_ready = torch.profiler.tensorboard_trace_handler(**trace_cfg)
            else:
                raise ValueError(
                    'trace_type should be "log_trace" or '
                    f'"tb_trace", but got {trace_type}'
                )
        else:
            raise ValueError(
                "``on_trace_ready`` should be a handler, or dict, or None, "
                f"but got {self.on_trace_ready}"
            )
        return _on_trace_ready

    def after_train_epoch(self, runner):
        """Determine if the content is exported."""
        # `after_train_epoch` will also be called in IterBasedTrainLoop.
        # Here we check `self._closed` to avoid exiting twice.
        if not self._closed:
            self._export_chrome_trace(runner)

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """profiler will call `step` method if it is not closed."""
        if not self._closed:
            self.profiler.step()
        if runner.iter == self.profile_times - 1 and not self.by_epoch:
            self._export_chrome_trace(runner)

    def _export_chrome_trace(self, runner):
        """Exporting content."""
        self._closed = True
        runner.logger.info("profiler may take a few minutes...")
        self.profiler.__exit__(None, None, None)
        if self.json_trace_path is not None:
            self.profiler.export_chrome_trace(self.json_trace_path)


@HOOKS.register_module()
class NPUProfilerHook(Hook):
    """NPUProfiler用于分析训练过程中的性能. 

    NPU Profiling 用于计算所有操作符的设备执行时间. 

    参数:
        begin (int): 进行性能分析的起始迭代次数. 默认为0. 
        end (int): 进行性能分析的结束迭代次数. 默认为1. 
        result_path (str): 保存性能分析结果文件的路径. 默认为'cann_profiling'. 
        exit_after_profiling (bool): 是否在性能分析后退出程序. 默认为True. 
        use_e2e_profiler (bool): 是否开启E2E性能分析, E2E性能分析结合了Pytorch层面和NPU层面的性能数据, 以端到端的方式分析模型性能的瓶颈, 无法展示详细内容, 仅作为辅助分析. 默认为False. 
        ge_profiling_to_std_out (bool): 是否开启GE性能分析, GE用于收集Assend设备主机端调度的性能分析数据. 默认为False. 

    """

    priority = "VERY_LOW"

    def __init__(
        self,
        *,
        begin: int = 0,
        end: int = 1,
        result_path: str = "cann_profiling",
        exit_after_profiling: bool = True,
        use_e2e_profiler: bool = False,
        ge_profiling_to_std_out: bool = False,
    ):
        try:
            import torch_npu
        except ImportError:
            raise ImportError("Failed to import torch_npu module")

        if begin >= end:
            raise ValueError(
                "The iteration to start profiling should not be greater"
                "than or equal to profile end"
            )

        self.begin = begin
        self.end = end
        self.result_path = result_path
        self.exit_after_profiling = exit_after_profiling

        if ge_profiling_to_std_out:
            os.environ["GE_PROFILING_TO_STD_OUT"] = "1"

        if not osp.exists(self.result_path):
            os.makedirs(self.result_path, exist_ok=True)

        self.profiler = torch_npu.npu.profile(
            self.result_path, use_e2e_profiler=use_e2e_profiler
        )

    @master_only
    def before_run(self, runner):
        if self.end > runner.max_iters:
            raise ValueError(
                "The profiling end iteration should not be greater"
                "than the max iteration"
            )

    @master_only
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if runner.iter == self.begin:
            self.profiler.__enter__()
            runner.logger.info("NPUProfiler starts profiling...")

    @master_only
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter == self.end - 1:
            runner.logger.info(
                "profiler may take a few minutes to" " save the profiling result."
            )
            self.profiler.__exit__(None, None, None)
            if self.exit_after_profiling:
                sys.exit()
