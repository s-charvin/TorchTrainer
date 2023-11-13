by_epoch = True
work_dir = "./work_dir/lightsernet"
experiment_name = "lightsernet"
model = dict(type="LightSerNet")


data_preprocessor = None

dataset_transforms = [
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
        win_length=16000 * 7,
        keys=["audio_data"],
    ),
    dict(
        type="PadCutAudio",
        sample_num=16000 * 7,
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

dataset = dict(
    type="IEMOCAP4C",
    root="/sdb/visitors2/SCW/data/IEMOCAP",
    enable_video=False,
    threads=0,
    serialize=False,
)

train_dataset = dict(
    type="IEMOCAP4C",
    root="/home/visitors2/SCW/TorchTrainer/work_dir/lightsernet/train/data.pkl",
    enable_video=False,
    threads=0,
    transforms=dataset_transforms,
)

val_dataset = dict(
    type="IEMOCAP4C",
    root="/home/visitors2/SCW/TorchTrainer/work_dir/lightsernet/val/data.pkl",
    enable_video=False,
    threads=0,
    transforms=dataset_transforms,
)

train_dataloader = dict(
    batch_size=64,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=200,
    val_begin=1,
    val_interval=1,
)
auto_scale_lr = None


optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.001,
        betas=(0.9, 0.999),
    ),
)
# 200 个 epoch, 学习率从 0.001 降到 1e-7，=, 每 50 个 epoch 降低 0.00025 / 200 = 1.25e-6
param_scheduler = [
    dict(
        type="CosineRestartParamScheduler",
        param_name="lr",
        periods=[50, 100, 150, 200],
        restart_weights=[1, 0.5, 0.25, 0.125],
        eta_min=1e-7,
        last_step=-1,
        by_epoch=by_epoch,
    ),
]

# param_scheduler = [
#     dict(
#         type="CosineLearningRateWithWarmRestartsParamScheduler",
#         param_name="lr",
#         warmup_epochs=40,
#         cosine_end_lr=1e-6,
#         warmup_start_lr=1e-8,
#         epochs=200,
#         last_step=-1,
#         by_epoch=by_epoch,
#     ),
# ]


val_dataloader = dict(
    batch_size=64,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)

val_cfg = dict(
    type="ValLoop",
)


val_evaluator = [
    dict(type="Accuracy"),
    dict(type="ConfusionMatrix"),
    dict(type="SingleLabelMetric"),
]

test_dataloader = dict(
    batch_size=32,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)

test_cfg = dict(
    type="TestLoop",
)

test_evaluator = [
    dict(type="Accuracy"),
    dict(type="ConfusionMatrix"),
    dict(type="SingleLabelMetric"),
]


custom_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    logger=dict(type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
)

resume = False
load_from = None

launcher = "none"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(
        mp_start_method="fork",
        opencv_num_threads=0,
    ),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=by_epoch)
log_level = "INFO"
visualizer = dict(
    type="Visualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
)

randomness = dict(seed=None)