import itertools
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from TorchTrainer.utils.dist import get_dist_info, sync_random_seed
from TorchTrainer.utils.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DefaultSampler(Sampler):
    """适用于分布式和非分布式环境的默认数据采样器, 用于生成顺序或随机索引的迭代器.
    - 支持分布式环境
    - 取整策略(向上取整/向下取整)
    """

    def __init__(
        self,
        dataset: Sized,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """构建索引索引迭代器."""

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[
                : self.total_size
            ]
        indices = indices[self.rank : self.total_size : self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch, 使得每个 epoch 使用不同的随机种子."""
        self.epoch = epoch


@DATA_SAMPLERS.register_module()
class InfiniteSampler(Sampler):
    """适用于基于迭代的训练器的无限数据采样器, 每次生成一个 mini-batch 的索引."""

    def __init__(
        self, dataset: Sized, shuffle: bool = True, seed: Optional[int] = None
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.size = len(dataset)
        self.indices = yield from itertools.islice(
            self._infinite_indices(), self.rank, None, self.world_size
        )

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        yield from self.indices

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        pass
