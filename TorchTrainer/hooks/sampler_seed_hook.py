from TorchTrainer.utils.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class DistSamplerSeedHook(Hook):
    """用于保证分布式训练中数据的一致性.
    通过设置 sampler 和 batch_sampler 的种子(seed), 以确保在每个训练周期(epoch)中数据的采样顺序是一致的.
    仅在分布式训练时有用, 且仅在 :obj:`EpochBasedRunner` 中有用, :obj:`IterBasedRunner` 可以通过 :obj:`IterLoader` 实现相同的目的.
    """

    priority = "NORMAL"

    def before_train_epoch(self, runner) -> None:
        """Set the seed for sampler and batch_sampler.

        Args:
            runner (Runner): The runner of the training process.
        """
        if hasattr(runner.train_loop.dataloader, "sampler") and hasattr(
            runner.train_loop.dataloader.sampler, "set_epoch"
        ):
            # In case the` _SingleProcessDataLoaderIter` has no sampler,
            # or data loader uses `SequentialSampler` in Pytorch.
            runner.train_loop.dataloader.sampler.set_epoch(runner.epoch)

        elif hasattr(runner.train_loop.dataloader, "batch_sampler") and hasattr(
            runner.train_loop.dataloader.batch_sampler.sampler, "set_epoch"
        ):
            # In case the` _SingleProcessDataLoaderIter` has no batch sampler.
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.train_loop.dataloader.batch_sampler.sampler.set_epoch(runner.epoch)
