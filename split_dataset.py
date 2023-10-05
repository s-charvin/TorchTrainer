import argparse
import logging
import os
import os.path as osp
from typing import Sequence, Union
from TorchTrainer.config import Config
from TorchTrainer.utils.logging import print_log
from TorchTrainer.utils.registry import RUNNERS, DATASETS, TRANSFORMS
from TorchTrainer.runner import Runner
from TorchTrainer.datasets import BaseDataset


def parse_args():
    parser = argparse.ArgumentParser(description="split a dataset")
    parser.add_argument("dataset_config", help="train config file path")
    parser.add_argument(
        "--work-dir", default=None, help="the dir to save logs and models"
    )
    parser.add_argument(
        "--indices",
        default=[0.8, 0.1, 0.1],
        type=Union[Sequence[int], Sequence[float], int],
        help="indices of dataset to be split",
    )
    parser.add_argument(
        "--names",
        default=["train", "val", "test"],
        type=Union[Sequence[str], str],
        help="names of sub datasets",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    dataset = cfg.get("dataset")

    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = osp.join(dataset.root, "sub_datasets")

    # split and save dataset
    dataset = DATASETS.build(dataset)
    sub_datasets = dataset.split(args.indices)
    for name, sub_dataset in zip(args.names, sub_datasets):
        sub_dataset.save(osp.join(work_dir, name, ".pkl"))


if __name__ == "__main__":
    main()
