import _init_paths

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
from TorchTrainer.datasets import IEMOCAP4C

from model import LightSerNet


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

    model = LightSerNet()


    pipeline = [
            dict(
                type="LoadAudioFromFile",
                backend="librosa",
                backend_args=None,
                to_float32=True,
            ),
            # dict(
            #     type="ResampleAudio",
            #     sample_rate=16000,
            #     keys=["audio_data"],
            # ),
            dict(
                type="TimeRandomSplit",
                win_length=16000*7,
                keys=["audio_data"],
            ),
            dict(
                type="PadCutAudio",
                sample_num=16000*7,
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
        ]


    dataset = DATASETS.build(    dict(
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
        transforms=pipeline,
    ))
    
    train_dataset, val_dataset = dataset.get_subset([0.8,0.2])
    test_dataset = None
    
    train_dataloader = dict(
        batch_size=32,
        num_workers=0,
        sampler=dict(type="DefaultSampler", shuffle=True),
        dataset=train_dataset,
    )

    val_dataloader = dict(
        batch_size=32,
        num_workers=0,
        sampler=dict(type="DefaultSampler", shuffle=True),
        dataset=val_dataset,
    )

    test_dataloader = dict(
    batch_size=32,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=test_dataset,
)

    evaluator = [
            dict(type="Accuracy"),
            dict(type="ConfusionMatrix"),
            dict(type="SingleLabelMetric"),
        ]

    train_cfg = dict(
        type="EpochBasedTrainLoop",
        max_epochs=200,
        val_begin=1,
        val_interval = 1,
    )

    val_cfg = dict(type='ValLoop')
    test_cfg = dict(type='TestLoop')

    param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]



    optim_wrapper = dict(
        type="OptimWrapper",
        optimizer=dict(
            type="AdamW",
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=0.001,
        ),
    )

    # # Default setting for scaling LR automatically
    # #   - `enable` means enable scaling LR automatically
    # #       or not by default.
    # #   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
    # auto_scale_lr = dict(enable=False, base_batch_size=16)

    visualizer = dict(type="Visualizer",
                      vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
                      name="visualizer",
            )

    log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)

    runner = Runner(
        model=model,
        work_dir="/sdb/visitors2/SCW/work_dir/LightSerNet-IEMOCAP4C-0.7-0.15-0.15",
        experiment_name = None,  # 实验名称
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=optim_wrapper,
        auto_scale_lr = None,
        param_scheduler = param_scheduler,

        val_dataloader=val_dataloader,
        val_cfg=val_cfg,
        val_evaluator=evaluator,

        test_dataloader = test_dataloader,
        test_cfg = test_cfg,
        test_evaluator = evaluator,

        default_hooks = None,
        load_from = None,
        resume = False,
        visualizer = visualizer,  # 可视化器

        launcher=args.launcher,
        log_processor = log_processor,
    )
    runner.train()


if __name__ == "__main__":
    main()
