import logging
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F


from torchTraining.utils.dist import (
    broadcast_object_list,
    collect_results,
    is_main_process,
)
from torchTraining.utils.fileio import dump
from torchTraining.utils.logging import print_log
from torchTraining.utils.registry import METRICS
from torchTraining.structures import BaseDataElement
from torchTraining.utils import is_str

from itertools import product
from typing import List, Optional, Sequence, Union


class BaseMetric(metaclass=ABCMeta):
    """Base class for a metric.

    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.

    A subclass of class:`BaseMetric` should assign a meaningful value to the
    class attribute `default_prefix`. See the argument `prefix` for details.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    default_prefix: Optional[str] = None

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        collect_dir: Optional[str] = None,
    ) -> None:
        if collect_dir is not None and collect_device != "cpu":
            raise ValueError(
                "`collec_dir` could only be configured when " "`collect_device='cpu'`"
            )

        self._dataset_meta: Union[None, dict] = None
        self.collect_device = collect_device
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        self.collect_dir = collect_dir

        if self.prefix is None:
            print_log(
                "The prefix is not set in metric class " f"{self.__class__.__name__}.",
                logger="current",
                level=logging.WARNING,
            )

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f"{self.__class__.__name__} got empty `self.results`. Please "
                "ensure that the processed results are properly added into "
                "`self.results` in `process` method.",
                logger="current",
                level=logging.WARNING,
            )

        if self.collect_device == "cpu":
            results = collect_results(
                self.results, size, self.collect_device, tmpdir=self.collect_dir
            )
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)
            # Add prefix to metric names
            if self.prefix:
                _metrics = {"/".join((self.prefix, k)): v for k, v in _metrics.items()}
            metrics = [_metrics]
        else:
            metrics = [None]

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]


@METRICS.register_module()
class DumpResults(BaseMetric):
    """Dump model predictions to a pickle file for offline evaluation.

    Args:
        out_file_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    def __init__(
        self,
        out_file_path: str,
        collect_device: str = "cpu",
        collect_dir: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, collect_dir=collect_dir)
        if not out_file_path.endswith((".pkl", ".pickle")):
            raise ValueError("The output file must be a pkl file.")
        self.out_file_path = out_file_path

    def process(self, data_batch: Any, predictions: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        self.results.extend(_to_cpu(predictions))

    def compute_metrics(self, results: list) -> dict:
        """dump the prediction results to a pickle file."""
        dump(results, self.out_file_path)
        print_log(f"Results has been saved to {self.out_file_path}.", logger="current")
        return {}


def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to("cpu")
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f"{type(value)} is not an available argument.")
    return value


def _precision_recall_f1_support(pred_positive, gt_positive, average):
    """calculate base classification task metrics, such as  precision, recall,
    f1_score, support."""
    average_options = ["micro", "macro", None]
    assert average in average_options, (
        "Invalid `average` argument, " f"please specify from {average_options}."
    )

    # ignore -1 target such as difficult sample that is not wanted
    # in evaluation results.
    # only for calculate multi-label without affecting single-label behavior
    ignored_index = gt_positive == -1
    pred_positive[ignored_index] = 0
    gt_positive[ignored_index] = 0

    class_correct = pred_positive & gt_positive
    if average == "micro":
        tp_sum = class_correct.sum()
        pred_sum = pred_positive.sum()
        gt_sum = gt_positive.sum()
    else:
        tp_sum = class_correct.sum(0)
        pred_sum = pred_positive.sum(0)
        gt_sum = gt_positive.sum(0)

    precision = tp_sum / torch.clamp(pred_sum, min=1).float() * 100
    recall = tp_sum / torch.clamp(gt_sum, min=1).float() * 100
    f1_score = (
        2
        * precision
        * recall
        / torch.clamp(precision + recall, min=torch.finfo(torch.float32).eps)
    )
    if average in ["macro", "micro"]:
        precision = precision.mean(0)
        recall = recall.mean(0)
        f1_score = f1_score.mean(0)
        support = gt_sum.sum(0)
    else:
        support = gt_sum
    return precision, recall, f1_score, support


@METRICS.register_module()
class Accuracy(BaseMetric):
    r"""用于单标签或多标签分类任务的准确性评估, 包含 top-k 准确度.

    Args:
        topk (int | Sequence[int]): 如果真实标签与最佳的 k 个预测之一匹配, 则将该样本视为正确预测。如果参数是元组, 则所有 top-k 准确度都将被计算并一起输出. 默认值为 1.
        thrs (Sequence[float | None] | float | None): 设置预测阈值. 如果是浮点数, 低于 thrs 值的预测被认为是错误预测的. None 意味着没有门槛. 如果是元组, 则将基于所有阈值计算并一起输出准确性. 默认为 0.
        collect_device (str): 设置在分布式训练期间收集不同训练进程中结果的设备类型. 必须是 "cpu" 或"gpu". 默认为 "cpu".
        prefix (str, optional): 将 prefix 添加到评估名称中, 避免不同评估器同名. 默认为 None, 使用 self.default_prefix.
    """
    default_prefix: Optional[str] = "accuracy"

    def __init__(
        self,
        topk: Union[int, Sequence[int]] = (1,),
        thrs: Union[float, Sequence[Union[float, None]], None] = 0.0,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(topk, int):
            self.topk = (topk,)
        else:
            self.topk = tuple(topk)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs,)
        else:
            self.thrs = tuple(thrs)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """处理 batch 内的数据样本, 存储在 "self.results" 中, 然后进行指标计算."""

        for data_sample in data_samples:
            result = dict()

            result["pred_score"] = data_sample["pred_label"]["score"].cpu()
            result["pred_label"] = data_sample["pred_label"]["label"].cpu()
            result["gt_label"] = data_sample["gt_label"]["label"].cpu()
            result["gt_score"] = data_sample["gt_label"]["score"].cpu()
            self.results.append(result)

    def compute_metrics(self, results: List):
        """计算准确度评估指标"""
        metrics = {}
        target = torch.cat([res["gt_label"] for res in results])
        if "pred_score" in results[0]:
            pred = torch.stack([res["pred_score"] for res in results])

            try:
                acc = self.calculate(pred, target, self.topk, self.thrs)
            except ValueError as e:
                raise ValueError(
                    str(e) + " Please check the `val_evaluator` and "
                    "`test_evaluator` fields in your config file."
                )

            multi_thrs = len(self.thrs) > 1
            for i, k in enumerate(self.topk):
                for j, thr in enumerate(self.thrs):
                    name = f"top{k}"
                    if multi_thrs:
                        name += "_no-thr" if thr is None else f"_thr-{thr:.2f}"
                    metrics[name] = acc[i][j].item()
        else:
            pred = torch.cat([res["pred_label"] for res in results])
            acc = self.calculate(pred, target, self.topk, self.thrs)
            metrics["top1"] = acc.item()

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        topk: Sequence[int] = (1,),
        thrs: Sequence[Union[float, None]] = (0.0,),
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the accuracy."""

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        num = pred.size(0)
        assert pred.size(0) == target.size(0), (
            f"The size of pred ({pred.size(0)}) doesn't match "
            f"the target ({target.size(0)})."
        )

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).float().sum(0, keepdim=True)
            acc = correct.mul_(100.0 / num)
            return acc
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = max(topk)

            if maxk > pred.size(1):
                raise ValueError(
                    f"Top-{maxk} accuracy is unavailable since the number of "
                    f"categories is {pred.size(1)}."
                )

            pred_score, pred_label = pred.topk(maxk, dim=1)
            pred_label = pred_label.t()
            correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
            results = []
            for k in topk:
                results.append([])
                for thr in thrs:
                    # Only prediction values larger than thr are counted
                    # as correct
                    _correct = correct
                    if thr is not None:
                        _correct = _correct & (pred_score.t() > thr)
                    correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
                    acc = correct_k.mul_(100.0 / num)
                    results[-1].append(acc)
            return results


@METRICS.register_module()
class SingleLabelMetric(BaseMetric):
    r"""用于单标签多分类任务的评估, 包含准确度, 召回率, F1-score.

    All metrics can be formulated use variables above:

    Args:
        thrs (Sequence[float | None] | float | None): 设置预测阈值. 如果是浮点数, 低于 thrs 值的预测被认为是错误预测的. None 意味着没有门槛. 如果是元组, 则将基于所有阈值计算并一起输出准确性. 默认为 0.
        items (Sequence[str]): 指定需要的评估指标， 默认为 ``('precision', 'recall', 'f1-score')``. 还支持 "support" .
        average (str | None): 设置混淆矩阵的计算模式, 默认为 "macro".
            - `"macro"`: 为每个类别计算指标, 并计算所有类别的平均值.
            - `"micro"`: 对所有类别进行混淆矩阵平均化, 并在平均混淆矩阵上计算指标.
            - `None`: 计算每个类别的指标并直接输出.
        num_classes (int, optional): 分类的类别数. 默认为None.
        collect_device (str): 设置在分布式训练期间收集不同训练进程中结果的设备类型. 必须是 "cpu" 或"gpu". 默认为 "cpu".
        prefix (str, optional): 将 prefix 添加到评估名称中, 避免不同评估器同名. 默认为 None, 使用 self.default_prefix.
    """
    default_prefix: Optional[str] = "single-label"

    def __init__(
        self,
        thrs: Union[float, Sequence[Union[float, None]], None] = 0.0,
        items: Sequence[str] = ("precision", "recall", "f1-score"),
        average: Optional[str] = "macro",
        num_classes: Optional[int] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs,)
        else:
            self.thrs = tuple(thrs)

        for item in items:
            assert item in ["precision", "recall", "f1-score", "support"], (
                f"The metric {item} is not supported by `SingleLabelMetric`,"
                ' please specify from "precision", "recall", "f1-score" and '
                '"support".'
            )
        self.items = tuple(items)
        self.average = average
        self.num_classes = num_classes

    def process(self, data_batch, data_samples: Sequence[dict]):
        """处理 batch 内的数据样本, 存储在 "self.results" 中, 然后进行指标计算."""

        for data_sample in data_samples:
            result = dict()

            result["pred_score"] = data_sample["pred_label"]["score"].cpu()
            result["pred_label"] = data_sample["pred_label"]["label"].cpu()
            result["num_classes"] = self.num_classes or data_sample.get("num_classes")
            assert (
                result["num_classes"] is not None
            ), "The `num_classes` must be specified if no `pred_score`."

            result["gt_label"] = data_sample["gt_label"]["label"].cpu()
            result["gt_score"] = data_sample["gt_label"]["score"].cpu()
            self.results.append(result)

    def compute_metrics(self, results: List):
        """ "计算准确度评估指标"""

        metrics = {}

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if "precision" in self.items:
                single_metrics["precision"] = precision
            if "recall" in self.items:
                single_metrics["recall"] = recall
            if "f1-score" in self.items:
                single_metrics["f1-score"] = f1_score
            if "support" in self.items:
                single_metrics["support"] = support
            return single_metrics

        # concat
        target = torch.cat([res["gt_label"] for res in results])
        if "pred_score" in results[0]:
            pred = torch.stack([res["pred_score"] for res in results])
            metrics_list = self.calculate(
                pred, target, thrs=self.thrs, average=self.average
            )

            multi_thrs = len(self.thrs) > 1
            for i, thr in enumerate(self.thrs):
                if multi_thrs:
                    suffix = "_no-thr" if thr is None else f"_thr-{thr:.2f}"
                else:
                    suffix = ""

                for k, v in pack_results(*metrics_list[i]).items():
                    metrics[k + suffix] = v
        else:
            pred = torch.cat([res["pred_label"] for res in results])
            res = self.calculate(
                pred,
                target,
                average=self.average,
                num_classes=results[0]["num_classes"],
            )
            metrics = pack_results(*res)

        result_metrics = dict()
        for k, v in metrics.items():
            if self.average is None:
                result_metrics[k + "_classwise"] = v.cpu().detach().tolist()
            elif self.average == "micro":
                result_metrics[k + f"_{self.average}"] = v.item()
            else:
                result_metrics[k] = v.item()

        return result_metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        thrs: Sequence[Union[float, None]] = (0.0,),
        average: Optional[str] = "macro",
        num_classes: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the precision, recall, f1-score and support."""
        average_options = ["micro", "macro", None]
        assert average in average_options, (
            "Invalid `average` argument, " f"please specify from {average_options}."
        )

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), (
            f"The size of pred ({pred.size(0)}) doesn't match "
            f"the target ({target.size(0)})."
        )

        if pred.ndim == 1:
            assert num_classes is not None, (
                "Please specify the `num_classes` if the `pred` is labels "
                "intead of scores."
            )
            gt_positive = F.one_hot(target.flatten(), num_classes)
            pred_positive = F.one_hot(pred.to(torch.int64), num_classes)
            return _precision_recall_f1_support(pred_positive, gt_positive, average)
        else:
            # For pred score, calculate on all thresholds.
            num_classes = pred.size(1)
            pred_score, pred_label = torch.topk(pred, k=1)
            pred_score = pred_score.flatten()
            pred_label = pred_label.flatten()

            gt_positive = F.one_hot(target.flatten(), num_classes)

            results = []
            for thr in thrs:
                pred_positive = F.one_hot(pred_label, num_classes)
                if thr is not None:
                    pred_positive[pred_score <= thr] = 0
                results.append(
                    _precision_recall_f1_support(pred_positive, gt_positive, average)
                )

            return results


@METRICS.register_module()
class ConfusionMatrix(BaseMetric):
    r"""用于单标签多分类任务的评估, 包含混淆矩阵.

    Args:
        num_classes (int, optional): 分类的类别数. 默认为None.
        collect_device (str): 设置在分布式训练期间收集不同训练进程中结果的设备类型. 必须是 "cpu" 或"gpu". 默认为 "cpu".
        prefix (str, optional): 将 prefix 添加到评估名称中, 避免不同评估器同名. 默认为 None, 使用 self.default_prefix.
    """
    default_prefix = "confusion_matrix"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device, prefix)

        self.num_classes = num_classes

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()

            result["pred_score"] = data_sample["pred_label"]["score"].cpu()
            result["pred_label"] = data_sample["pred_label"]["label"].cpu()

            self.num_classes = data_sample.get("num_classes") or result[
                "pred_score"
            ].size(0)

            assert (
                self.num_classes is not None
            ), "The `num_classes` must be specified if no `pred_score`."

            result["gt_label"] = data_sample["gt_label"]["label"].cpu()
            result["gt_score"] = data_sample["gt_label"]["score"].cpu()

            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        pred_labels = []
        gt_labels = []
        for result in results:
            pred_labels.append(result["pred_label"])
            gt_labels.append(result["gt_label"])
        confusion_matrix = ConfusionMatrix.calculate(
            torch.cat(pred_labels), torch.cat(gt_labels), num_classes=self.num_classes
        )
        return {"result": confusion_matrix}

    @staticmethod
    def calculate(pred, target, num_classes=None) -> dict:
        """Calculate the confusion matrix for single-label task."""
        pred = to_tensor(pred)
        target_label = to_tensor(target).int()

        assert pred.size(0) == target_label.size(0), (
            f"The size of pred ({pred.size(0)}) doesn't match "
            f"the target ({target_label.size(0)})."
        )
        assert target_label.ndim == 1

        if pred.ndim == 1:
            assert num_classes is not None, (
                "Please specify the `num_classes` if the `pred` is labels "
                "intead of scores."
            )
            pred_label = pred
        else:
            num_classes = num_classes or pred.size(1)
            pred_label = torch.argmax(pred, dim=1).flatten()

        with torch.no_grad():
            indices = num_classes * target_label + pred_label
            matrix = torch.bincount(indices, minlength=num_classes**2)
            matrix = matrix.reshape(num_classes, num_classes)

        return matrix

    @staticmethod
    def plot(
        confusion_matrix: torch.Tensor,
        include_values: bool = False,
        cmap: str = "viridis",
        classes: Optional[List[str]] = None,
        colorbar: bool = True,
        show: bool = True,
    ):
        """Draw a confusion matrix by matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        num_classes = confusion_matrix.size(0)

        im_ = ax.imshow(confusion_matrix, interpolation="nearest", cmap=cmap)
        text_ = None
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

        if include_values:
            text_ = np.empty_like(confusion_matrix, dtype=object)

            # print text with appropriate color depending on background
            thresh = (confusion_matrix.max() + confusion_matrix.min()) / 2.0

            for i, j in product(range(num_classes), range(num_classes)):
                color = cmap_max if confusion_matrix[i, j] < thresh else cmap_min

                text_cm = format(confusion_matrix[i, j], ".2g")
                text_d = format(confusion_matrix[i, j], "d")
                if len(text_d) < len(text_cm):
                    text_cm = text_d

                text_[i, j] = ax.text(
                    j, i, text_cm, ha="center", va="center", color=color
                )

        display_labels = classes or np.arange(num_classes)

        if colorbar:
            fig.colorbar(im_, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_ylim((num_classes - 0.5, -0.5))
        # Automatically rotate the x labels.
        fig.autofmt_xdate(ha="center")

        if show:
            plt.show()
        return fig
