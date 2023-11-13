import torch
from torch import Tensor, nn
import torch.nn.functional as F
from TorchTrainer.utils.registry import DATASETS, MODELS
from TorchTrainer.model import BaseModel
from TorchTrainer.structures import ClsDataSample

from . import components


@MODELS.register_module()
class LightSerNet(BaseModel):
    """
    paper: Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition
    """

    def __init__(self) -> None:
        super().__init__()
        self.path1 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(11, 1), stride=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path2 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(1, 9), stride=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.path3 = nn.Sequential(
            components.Conv2dSame(
                in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.conv_extractor = components._get_Conv2d_extractor(
            in_channels=32 * 3,
            shapes=[  # [channel:int, kernel:list|int, stride:list|int, norm:str|None, pool:str|None, pool_kernel:list|int]
                [64, [3, 3], 1, "bn", "ap", [2, 2]],
                [96, [3, 3], 1, "bn", "ap", [2, 2]],
                [128, [3, 3], 1, "bn", "ap", [2, 1]],
                [160, [3, 3], 1, "bn", "ap", [2, 1]],
                [320, [1, 1], 1, "bn", "gap", 1],
            ],
            bias=False,
        )
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(in_features=320, out_features=4)

    def _forward(self, x: Tensor, feature_lens=None) -> Tensor:
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        x = torch.cat([out1, out2, out3], dim=1).contiguous()

        x = self.conv_extractor(x)
        x = self.dropout(x.squeeze(2).squeeze(2))
        x = self.last_linear(x)

        return x

    def forward(self, inputs, data_samples, mode):
        data_sample = data_samples[0]
        assert (
            "gt_label" in data_sample
            and getattr(data_sample.gt_label, "label", None) is not None
        ), "gt_label must be provided"

        targets = torch.stack([i.gt_label.label for i in data_samples]).squeeze(1)
        mfcc = inputs["mfcc_data"]

        if mfcc.dim() == 3:
            mfcc = mfcc.unsqueeze(1)
        cls_score = self._forward(mfcc)

        if mode == "loss":
            return {"loss": F.cross_entropy(cls_score, targets)}
        elif mode == "predict":
            pred_scores = F.softmax(cls_score, dim=1)
            pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()
            out_data_samples = []
            if data_samples is None:
                data_samples = [None for _ in range(pred_scores.size(0))]
            for data_sample, score, label in zip(
                data_samples, pred_scores, pred_labels
            ):
                if data_sample is None:
                    data_sample = ClsDataSample()
                data_sample.set_pred_score(score).set_pred_label(label)
                out_data_samples.append(data_sample)
            return out_data_samples
