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

from .model import LightSerNet


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
        type="IEMOCAP4C",
        root="/sdb/visitors2/SCW/data/IEMOCAP",
        enable_video=False,
        threads=0,
        filter=dict(
            replace=dict(
                label=("excited", "happy"),
            ),
            drop=dict(
                label=[
                    "other",
                    "xxx",
                    "frustrated",
                    "disgusted",
                    "fearful",
                    "surprised",
                ],
            ),
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
            dict(type="PackAudioClsInputs", keys=["mfcc_data"]),
        ],
    )

    model = LightSerNet()
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
