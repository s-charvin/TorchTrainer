
from typing import Any, Iterator, List, Optional, Sequence, Union

from TorchTrainer.utils.registry import EVALUATOR, METRICS
from TorchTrainer.datasets import input_collate
from TorchTrainer.structures import BaseDataElement
from .metric import BaseMetric


@EVALUATOR.register_module()
class Evaluator:
    """用于组合多个 BaseMetric 实例的包装器类
    """
    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        self._dataset_meta: Optional[dict] = None
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
        self.metrics: List[BaseMetric] = []
        for metric in metrics:
            if isinstance(metric, dict):
                self.metrics.append(METRICS.build(metric))
            else:
                self.metrics.append(metric)

    @property
    def dataset_meta(self) -> Optional[dict]:
        """数据集的元信息."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """将数据集元信息设置到 evaluator 及其 metrics 中."""
        self._dataset_meta = dataset_meta
        for metric in self.metrics:
            metric.dataset_meta = dataset_meta

    def process(self,
                data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """将 BaseDataSample 转换为字典, 并调用每个 metric 的处理方法

        Args:
            data_samples (Sequence[BaseDataElement]): 模型的预测结果和验证集的真实标签.
            data_batch (Any, optional): 数据加载器中的一批数据.
        """
        _data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, BaseDataElement):
                _data_samples.append(data_sample.to_dict())
            else:
                _data_samples.append(data_sample)

        for metric in self.metrics:
            metric.process(data_batch, _data_samples)

    def evaluate(self, size: int) -> dict:
        """调用每个指标的 evaluate 方法, 并收集 metrics 字典

        Args:
            size (int): 整个验证数据集的长度. 当 batch 大小 > 1 时, 
                数据加载器可能会填充一些数据样本以确保所有 rank 都具有相同的数据集切片长度.
                collect_results 函数将根据此大小删除填充数据.

        Returns:
            dict: 所有指标的评估结果. key 是 metric 的名称, value 是相应的处理结果.
        """
        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # 检查 metric 名称冲突
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')
            metrics.update(_results)
        return metrics

    def offline_evaluate(self,
                         data_samples: Sequence,
                         data: Optional[Sequence] = None,
                         chunk_size: int = 1):
        """在给定的数据上离线评估转储的预测.

        Args:
            data_samples (Sequence): 模型和验证集的所有预测结果和真实标签.
            data (Sequence, optional): 验证集的所有数据.
            chunk_size (int): 要处理的数据样本和预测的数量.
        """

        # support chunking iterable objects
        def get_chunks(seq: Iterator, chunk_size=1):
            stop = False
            while not stop:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(next(seq))
                    except StopIteration:
                        stop = True
                        break
                if chunk:
                    yield chunk

        if data is not None:
            assert len(data_samples) == len(data), (
                'data_samples and data should have the same length, but got '
                f'data_samples length: {len(data_samples)} '
                f'data length: {len(data)}')
            data = get_chunks(iter(data), chunk_size)

        size = 0
        for output_chunk in get_chunks(iter(data_samples), chunk_size):
            if data is not None:
                data_chunk = input_collate(next(data))
            else:
                data_chunk = None
            size += len(output_chunk)
            self.process(output_chunk, data_chunk)
        return self.evaluate(size)
