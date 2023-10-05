from TorchTrainer.utils.dist import all_reduce_params, is_distributed
from TorchTrainer.utils.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class SyncBuffersHook(Hook):
    """在每个训练周期结束时同步模型缓冲区, 例如 BN 中的 running_mean 和 running_var. 
    它通过调用 all_reduce_params 函数来执行模型缓冲区的全局平均值操作, 以确保在分布式训练中所有进程的模型缓冲区保持一致. 这样可以避免在分布式训练中由于进程间通信造成的模型不一致性问题. 
    在验证周期之前, 如果在上一个训练周期结束时没有同步模型缓冲区, SyncBuffersHook 还会在每个验证周期之前执行一次同步操作. 
"""

    priority = "NORMAL"

    def __init__(self) -> None:
        self.distributed = is_distributed()
        # A flag to mark whether synchronization has been done in
        # after_train_epoch
        self.called_in_train = False

    def before_val_epoch(self, runner) -> None:
        """All-reduce model buffers before each validation epoch.

        Synchronize the buffers before each validation if they have not been
        synchronized at the end of the previous training epoch. This method
        will be called when using IterBasedTrainLoop.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.distributed:
            if not self.called_in_train:
                all_reduce_params(runner.model.buffers(), op="mean")
            self.called_in_train = False

    def after_train_epoch(self, runner) -> None:
        """All-reduce model buffers at the end of each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.distributed:
            all_reduce_params(runner.model.buffers(), op="mean")
            self.called_in_train = True
