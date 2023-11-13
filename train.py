import os.path as osp
import sys

this_dir = osp.dirname(__file__)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

import argparse
import logging
import os

import copy

from TorchTrainer.config import Config
from TorchTrainer.utils.logging import print_log
from TorchTrainer.utils.registry import RUNNERS
from TorchTrainer.runner import Runner
from TorchTrainer.utils.dl_utils import setup_cache_size_limit_of_dynamo
from TorchTrainer.datasets import BaseDataset

import projects


def parse_args():
    parser = argparse.ArgumentParser(description="默认训练脚本")
    parser.add_argument("config", help="训练配置文件的路径")
    parser.add_argument("--work-dir", help="训练日志和模型的保存路径")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="训练时是否使用自动混合精度模式",
    )
    parser.add_argument("--auto-scale-lr", action="store_true", help="训练时是否自动缩放学习率")
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="训练时是否从断点恢复训练, 默认为auto, 会自动从上次的断点恢复训练",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="训练时使用的分布式训练方式",
    )
    parser.add_argument(
        "--local_rank", "--local-rank", type=int, default=0, help="本地进程的rank"
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    # 加载训练配置文件
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "训练开启自动混合精度模式",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert (
                optim_wrapper == "OptimWrapper"
            ), f"`--amp` 只能在使用 `OptimWrapper` 时使用, 但是当前使用的是{optim_wrapper}"
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if (
            "auto_scale_lr" in cfg
            and "enable" in cfg.auto_scale_lr
            and "base_batch_size" in cfg.auto_scale_lr
        ):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                "训练配置文件中没有找到 `auto_scale_lr` 或者 `auto_scale_lr.enable` 或者 `auto_scale_lr.base_batch_size`"
            )

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    cfg = copy.deepcopy(cfg)
    runner = Runner(
        work_dir=cfg.get("work_dir", "./work_dirs"),
        experiment_name=cfg.get("experiment_name"),
        model=cfg.get("model"),
        data_preprocessor=cfg.get("data_preprocessor"),
        train_dataloader=cfg.get("train_dataloader"),
        optim_wrapper=cfg.get("optim_wrapper"),
        param_scheduler=cfg.get("param_scheduler"),
        auto_scale_lr=cfg.get("auto_scale_lr"),
        train_cfg=cfg.get("train_cfg"),
        val_dataloader=cfg.get("val_dataloader"),
        val_cfg=cfg.get("val_cfg"),
        val_evaluator=cfg.get("val_evaluator"),
        test_dataloader=cfg.get("test_dataloader"),
        test_cfg=cfg.get("test_cfg"),
        test_evaluator=cfg.get("test_evaluator"),
        custom_hooks=cfg.get("hooks"),
        resume=cfg.get("resume", False),
        load_from=cfg.get("load_from"),
        launcher=cfg.get("launcher", "none"),
        env_cfg=cfg.get("env_cfg"),
        log_processor=cfg.get("log_processor"),
        log_level=cfg.get("log_level", "INFO"),
        visualizer=cfg.get("visualizer"),
        randomness=cfg.get("randomness", dict(seed=None)),
        cfg=cfg,
    )

    # start training
    runner.train()


if __name__ == "__main__":
    main()
    # python ./train.py ./projects/lightsernet/config.py --work-dir ./work_dir/lightsernet --resume auto --launcher pytorch --local-rank 0
