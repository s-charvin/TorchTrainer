import argparse

import torch
from torch.optim import SGD
import torch.nn.functional as F
import torchvision


from TorchTrainer.utils.registry import DATASETS
from TorchTrainer.model import BaseModel
from TorchTrainer.runner import Runner
from TorchTrainer.evaluator import BaseMetric
from TorchTrainer.structures import label_data, ClsDataSample


class ResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(num_classes=7)
        average_weight = torch.mean(self.resnet.conv1.weight, dim=1, keepdim=True)
        self.resnet.conv1 = torch.nn.Conv2d(
            1,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias,
        )
        self.resnet.conv1.weight.data = average_weight

    def forward(self, inputs, data_samples, mode):
        data_sample = data_samples[0]
        assert (
            "gt_label" in data_sample
            and getattr(data_sample.gt_label, "label", None) is not None
        ), "gt_label must be provided"

        targets = torch.stack([i.gt_label.label for i in data_samples]).squeeze(1)
        mfcc = inputs["mfcc_data"][0]

        if mfcc.dim() == 3:
            mfcc = mfcc.unsqueeze(1)
        cls_score = self.resnet(mfcc)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset = dict(
        type="EMODB",
        root="/sdb/visitors2/SCW/data/EMODB",
        filter=dict(
            # replace=dict(label=("bored", "sad")),
            # dropna=["fearful"],
            # query="gender == 1",
        ),
        transforms=[
            dict(
                type="LoadAudioFromFile",
                backend="librosa",
                backend_args=None,
                to_float32=True,
            ),
            dict(
                type="ResampleAudio",
                sample_rate=16000,
                keys=["audio_data"],
            ),
            dict(
                type="TimeRandomSplit",
                win_length=16000,
                keys=["audio_data"],
            ),
            dict(
                type="PadCutAudio",
                sample_num=16000,
                pad_mode="random",
                keys=["audio_data"],
            ),
            dict(
                type="MFCC",
                keys=["audio_data"],
                override=False,
                backend="librosa",
                n_mfcc=40,
                dct_type=2,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                window="hann",
                center=True,
                pad_mode="reflect",
                power=2.0,
                n_mels=128,
                fmin=80.0,
                fmax=7600.0,
                norm="slaney",
                # torchaudio parameters
                lifter=0,
                log_mels=True,
                pad=0,
                # librosa parameters
                normalized=True,
                onesided=True,
            ),
            dict(
                type="MelSpectrogram",
                keys=["audio_data"],
                override=False,
                backend="librosa",
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                f_min=80.0,
                f_max=7600.0,
                pad=0,
                n_mels=128,
                power=2.0,
                normalized=True,
                center=True,
                pad_mode="reflect",
                onesided=True,
            ),
            dict(type="PackAudioClsInputs", keys=["mfcc_data"]),
        ],
    )

    model = ResNet50()
    dataloader = dict(
        batch_size=32,
        num_workers=0,
        sampler=dict(type="DefaultSampler", shuffle=True),
        dataset=dataset,
    )
    evaluator = dict(type="Accuracy")

    optim_wrapper = dict(
        type="OptimWrapper",
        optimizer=dict(
            type="AdamW",
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        ),
    )

    train_cfg = dict(
        type="IterBasedTrainLoop",
        max_iters=10000,
        val_interval=100,
    )
    # dict(by_epoch=True, max_epochs=2, val_interval=1)

    default_hooks = dict(
        logger=dict(type="LoggerHook", interval=50),
        checkpoint=dict(
            type="CheckpointHook",
            by_epoch=False,
            save_last=True,
            interval=50,
            max_keep_ckpts=5,
        ),
    )

    vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
    visualizer = dict(
        type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
    )
    log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)
    auto_scale_lr = dict(base_batch_size=64)
    runner = Runner(
        model=model,
        work_dir="./work_dir",
        train_dataloader=dataloader,
        optim_wrapper=optim_wrapper,
        train_cfg=train_cfg,
        val_dataloader=dataloader,
        val_cfg=dict(),
        val_evaluator=[
            dict(type="Accuracy"),
            dict(type="ConfusionMatrix"),
            dict(type="SingleLabelMetric"),
        ],
        launcher=args.launcher,
    )
    runner.train()


if __name__ == "__main__":
    main()
