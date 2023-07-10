from multiprocessing.reduction import ForkingPickler
from numbers import Number
from typing import Sequence, Union

import numpy as np
import torch
from torchTraining.structures import BaseDataElement, LabelData
from torchTraining.utils import is_str


def format_label(value: Union[torch.Tensor, np.ndarray, Sequence, int]) -> torch.Tensor:
    """Convert various python types to label-format tensor."""

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f"Type {type(value)} is not an available label type.")
    assert value.ndim == 1, f"The dims of value should be 1, but got {value.ndim}."

    return value


def format_score(value: Union[torch.Tensor, np.ndarray, Sequence, int]) -> torch.Tensor:
    """Convert various python types to score-format tensor."""

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f"Type {type(value)} is not an available label type.")
    assert value.ndim == 1, f"The dims of value should be 1, but got {value.ndim}."

    return value


class ClsDataSample(BaseDataElement):
    def set_gt_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> "ClsDataSample":
        """Set label of ``gt_label``."""
        label_data = getattr(self, "_gt_label", LabelData())
        label_data.label = format_label(value)
        self.gt_label = label_data
        return self

    def set_gt_score(self, value: torch.Tensor) -> "ClsDataSample":
        """Set score of ``gt_label``."""
        label_data = getattr(self, "_gt_label", LabelData())
        label_data.score = format_score(value)
        if hasattr(self, "num_classes"):
            assert len(label_data.score) == self.num_classes, (
                f"The length of score {len(label_data.score)} should be "
                f"equal to the num_classes {self.num_classes}."
            )
        else:
            self.set_field(
                name="num_classes", value=len(label_data.score), field_type="metainfo"
            )
        self.gt_label = label_data
        return self

    def set_pred_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> "ClsDataSample":
        """Set label of ``pred_label``."""
        label_data = getattr(self, "_pred_label", LabelData())
        label_data.label = format_label(value)
        self.pred_label = label_data
        return self

    def set_pred_score(self, value: torch.Tensor) -> "ClsDataSample":
        """Set score of ``pred_label``."""
        label_data = getattr(self, "_pred_label", LabelData())
        label_data.score = format_score(value)
        if hasattr(self, "num_classes"):
            assert len(label_data.score) == self.num_classes, (
                f"The length of score {len(label_data.score)} should be "
                f"equal to the num_classes {self.num_classes}."
            )
        else:
            self.set_field(
                name="num_classes", value=len(label_data.score), field_type="metainfo"
            )
        self.pred_label = label_data
        return self

    @property
    def gt_label(self):
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData):
        self.set_field("_gt_label", value, dtype=LabelData)

    @gt_label.deleter
    def gt_label(self):
        del self._gt_label

    @property
    def pred_label(self):
        return self._pred_label

    @pred_label.setter
    def pred_label(self, value: LabelData):
        self.set_field("_pred_label", value, dtype=LabelData)

    @pred_label.deleter
    def pred_label(self):
        del self._pred_label


class MultiTaskClsDataSample(BaseDataElement):
    @property
    def tasks(self):
        return self._data_fields
