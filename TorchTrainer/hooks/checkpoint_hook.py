import hashlib
import logging
import os.path as osp
import pickle
from collections import deque
from math import inf
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from TorchTrainer.utils.dist import is_main_process, master_only
from TorchTrainer.utils.fileio import get_file_backend
from TorchTrainer.utils.logging import print_log
from TorchTrainer.utils.registry import HOOKS
from TorchTrainer.utils import is_list_of, is_seq_of
from .hook import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class CheckpointHook(Hook):
    """定期保存检查点.

    参数:
        interval (int): 保存周期. 如果 ``by_epoch=True``, 则表示按照 epoch 保存, 否则表示按照迭代次数保存. 默认值为 -1, 表示“从不保存”.
        by_epoch (bool): 是否按照 epoch 保存检查点. 默认值为 True.
        save_optimizer (bool): 是否保存优化器的 state_dict 到检查点中. 通常用于恢复实验. 默认值为 True.
        save_param_scheduler (bool): 是否保存 param_scheduler 的 state_dict 到检查点中. 通常用于恢复实验. 默认值为 True.
        out_dir (str, Path, Optional): 保存检查点的根目录. 默认值为 None.
            如果未指定, 则默认使用 ``runner.work_dir``.
            如果指定了 ``out_dir``, 则将其与 ``runner.work_dir`` 的最后一级目录连接起来作为保存路径.
            ``./tmp`` +  ``./work_dir/cur_exp`` == ``./tmp/cur_exp`` 中.
        max_keep_ckpts (int): 最多保存的新的检查点数量. 默认值为 -1, 表示不限制数量.
        save_last (bool): 是否强制保存最后一个检查点, 而不考虑保存周期. 默认值为 True.
        save_best (str, List[str], optional): 根据指定的度量指标, 它将在评估过程中衡量最佳检查点并保存. 默认值为 None.
            关于最佳检查点的信息将保存在 ``runner.message_hub`` 中, 以保留最佳得分值和最佳检查点路径, 在恢复检查点时也会加载这些信息.
            如果 ``save_best`` 是 ``auto``, 则将使用返回的 ``OrderedDict`` 结果的第一个键.
        rule (str, List[str], optional): 最佳得分的比较规则. 如果设置为 None, 它将推断一个合理的规则.
            含有 'acc'、'top' 等关键字的键将推断为 'greater' 规则.
            含有 'loss' 等关键字的键将推断为 'less' 规则.
            如果 ``save_best`` 是度量指标列表, 而 ``rule`` 是字符串, 则所有度量指标都将共享比较规则.
            如果 ``save_best`` 和 ``rule`` 都是列表, 则它们的长度必须相同,
            并且 ``save_best`` 中的度量指标将使用 ``rule`` 中对应的比较规则.
            选项是 'greater'、'less'、None 和包含 'greater' 和 'less' 的列表.
            默认值为 None.
        greater_keys (List[str], optional): 自定义推断为 'greater' 比较规则的度量指标键. 默认值为 None.
            如果为 ``None``, 将使用 _default_greater_keys.
        less_keys (List[str], optional): 自定义推断为 'less' 比较规则的度量指标键. 默认值为 None.
            如果为 ``None``, 将使用 _default_less_keys.
        filename_tmpl (str, optional): 指示检查点名称的字符串模板. 默认值为 None, 表示分别使用 "epoch_{}.pth" 或 "iter_{}.pth".
            如果指定了, 必须包含一个且仅一个 "{}", 它将被替换为 ``epoch + 1`` (如果 ``by_epoch=True``),
            否则将被替换为 ``iteration + 1``.
        backend_args (dict, optional): 实例化 uri 对应后端的前缀的参数. 默认值为 None.
        published_keys (str, List[str], optional): 自动在训练后发布具有列表中键的模型. 默认值为 None.
            如果 ``save_last`` 是 ``True`` 或 ``save_best`` 不是 ``None``.
    """

    out_dir: str

    priority = "VERY_LOW"

    # logic to save best checkpoints
    # Since the key for determining greater or less is related to the
    # downstream tasks, downstream repositories may need to overwrite
    # the following inner variables accordingly.

    rule_map = {"greater": lambda x, y: x > y, "less": lambda x, y: x < y}
    init_value_map = {"greater": -inf, "less": inf}
    _default_greater_keys = [
        "acc",
        "top",
        "AR@",
        "auc",
        "precision",
        "mAP",
        "mDice",
        "mIoU",
        "mAcc",
        "aAcc",
    ]
    _default_less_keys = ["loss"]

    def __init__(
        self,
        interval: int = -1,
        by_epoch: bool = True,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        out_dir: Optional[Union[str, Path]] = None,
        max_keep_ckpts: int = -1,
        save_last: bool = True,
        save_best: Union[str, List[str], None] = None,
        rule: Union[str, List[str], None] = None,
        greater_keys: Optional[Sequence[str]] = None,
        less_keys: Optional[Sequence[str]] = None,
        filename_tmpl: Optional[str] = None,
        backend_args: Optional[dict] = None,
        published_keys: Union[str, List[str], None] = None,
        **kwargs,
    ) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_param_scheduler = save_param_scheduler
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs

        self.backend_args = backend_args

        if filename_tmpl is None:
            if self.by_epoch:
                self.filename_tmpl = "epoch_{}.pth"
            else:
                self.filename_tmpl = "iter_{}.pth"
        else:
            self.filename_tmpl = filename_tmpl

        # save best logic
        assert (
            isinstance(save_best, str)
            or is_list_of(save_best, str)
            or (save_best is None)
        ), (
            '"save_best" should be a str or list of str or None, '
            f"but got {type(save_best)}"
        )

        if isinstance(save_best, list):
            if "auto" in save_best:
                assert (
                    len(save_best) == 1
                ), 'Only support one "auto" in "save_best" list.'
            assert len(save_best) == len(
                set(save_best)
            ), 'Find duplicate element in "save_best".'
        else:
            # convert str to list[str]
            if save_best is not None:
                save_best = [save_best]
        self.save_best = save_best

        # rule logic
        assert isinstance(rule, str) or is_list_of(rule, str) or (rule is None), (
            '"rule" should be a str or list of str or None, ' f"but got {type(rule)}"
        )
        if isinstance(rule, list):
            # check the length of rule list
            assert len(rule) in [1, len(self.save_best)], (
                'Number of "rule" must be 1 or the same as number of '
                f'"save_best", but got {len(rule)}.'
            )
        else:
            # convert str/None to list
            rule = [rule]

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys,)
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys,)
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys

        if self.save_best is not None:
            self.is_better_than: Dict[str, Callable] = dict()
            self._init_rule(rule, self.save_best)
            if len(self.key_indicators) == 1:
                self.best_ckpt_path: Optional[str] = None
            else:
                self.best_ckpt_path_dict: Dict = dict()

        # published keys
        if not (
            isinstance(published_keys, str)
            or is_seq_of(published_keys, str)
            or published_keys is None
        ):
            raise TypeError(
                '"published_keys" should be a str or a sequence of str or '
                f"None, but got {type(published_keys)}"
            )

        if isinstance(published_keys, str):
            published_keys = [published_keys]
        elif isinstance(published_keys, (list, tuple)):
            assert len(published_keys) == len(
                set(published_keys)
            ), 'Find duplicate elements in "published_keys".'
        self.published_keys = published_keys

        self.last_ckpt = None

    def before_train(self, runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        self.file_backend = get_file_backend(
            self.out_dir, backend_args=self.backend_args
        )

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_backend.join_path(self.out_dir, basename)

        runner.logger.info(f"Checkpoints will be saved to {self.out_dir}.")

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                if "best_ckpt" not in runner.message_hub.runtime_info:
                    self.best_ckpt_path = None
                else:
                    self.best_ckpt_path = runner.message_hub.get_info("best_ckpt")
            else:
                for key_indicator in self.key_indicators:
                    best_ckpt_name = f"best_ckpt_{key_indicator}"
                    if best_ckpt_name not in runner.message_hub.runtime_info:
                        self.best_ckpt_path_dict[key_indicator] = None
                    else:
                        self.best_ckpt_path_dict[
                            key_indicator
                        ] = runner.message_hub.get_info(best_ckpt_name)

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = []
            if "keep_ckpt_ids" in runner.message_hub.runtime_info:
                keep_ckpt_ids = runner.message_hub.get_info("keep_ckpt_ids")

                while len(keep_ckpt_ids) > self.max_keep_ckpts:
                    step = keep_ckpt_ids.pop(0)
                    if is_main_process():
                        path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(step)
                        )
                        if self.file_backend.isfile(path):
                            self.file_backend.remove(path)
                        elif self.file_backend.isdir(path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(path)

            self.keep_ckpt_ids: deque = deque(keep_ckpt_ids, self.max_keep_ckpts)

    def after_train_epoch(self, runner) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval) or (
            self.save_last and self.is_last_train_epoch(runner)
        ):
            runner.logger.info(f"Saving checkpoint at {runner.epoch + 1} epochs")
            self._save_checkpoint(runner)

    def after_val_epoch(self, runner, metrics):
        """Save the checkpoint and synchronize buffers after each evaluation
        epoch.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics
        """
        if len(metrics) == 0:
            runner.logger.warning(
                "Since `metrics` is an empty dict, the behavior to save "
                "the best checkpoint will be skipped in this evaluation."
            )
            return

        self._save_best_checkpoint(runner, metrics)

    def after_train(self, runner) -> None:
        """Publish the checkpoint after training.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.published_keys is None:
            return

        if self.save_last and self.last_ckpt is not None:
            self._publish_model(runner, self.last_ckpt)

        if getattr(self, "best_ckpt_path", None) is not None:
            self._publish_model(runner, str(self.best_ckpt_path))
        if getattr(self, "best_ckpt_path_dict", None) is not None:
            for best_ckpt in self.best_ckpt_path_dict.values():
                self._publish_model(runner, best_ckpt)

    @master_only
    def _publish_model(self, runner, ckpt_path: str) -> None:
        """Remove unnecessary keys from ckpt_path and save the new checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            ckpt_path (str): The checkpoint path that ought to be published.
        """
        from TorchTrainer.runner import save_checkpoint
        from TorchTrainer.runner.checkpoint import _load_checkpoint

        checkpoint = _load_checkpoint(ckpt_path)
        assert self.published_keys is not None
        removed_keys = []
        for key in list(checkpoint.keys()):
            if key not in self.published_keys:
                removed_keys.append(key)
                checkpoint.pop(key)
        if removed_keys:
            print_log(
                f"Key {removed_keys} will be removed because they are not "
                "found in published_keys. If you want to keep them, "
                f"please set `{removed_keys}` in published_keys",
                logger="current",
            )
        checkpoint_data = pickle.dumps(checkpoint)
        sha = hashlib.sha256(checkpoint_data).hexdigest()
        final_path = osp.splitext(ckpt_path)[0] + f"-{sha[:8]}.pth"
        save_checkpoint(checkpoint, final_path)
        print_log(
            f"The checkpoint ({ckpt_path}) is published to " f"{final_path}.",
            logger="current",
        )

    def _save_checkpoint_with_step(self, runner, step, meta):
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step)
                        )

                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info(
                    "keep_ckpt_ids", list(self.keep_ckpt_ids)
                )

        ckpt_filename = self.filename_tmpl.format(step)
        self.last_ckpt = self.file_backend.join_path(self.out_dir, ckpt_filename)
        runner.message_hub.update_info("last_ckpt", self.last_ckpt)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args,
        )

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return

        save_file = osp.join(runner.work_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(self.last_ckpt)

    def _save_checkpoint(self, runner) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            step = runner.epoch + 1
            meta = dict(epoch=step, iter=runner.iter)
        else:
            step = runner.iter + 1
            meta = dict(epoch=runner.epoch, iter=step)

        self._save_checkpoint_with_step(runner, step, meta=meta)

    def _save_best_checkpoint(self, runner, metrics) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics.
        """
        if not self.save_best:
            return

        if self.by_epoch:
            ckpt_filename = self.filename_tmpl.format(runner.epoch)
            cur_type, cur_time = "epoch", runner.epoch
        else:
            ckpt_filename = self.filename_tmpl.format(runner.iter)
            cur_type, cur_time = "iter", runner.iter

        meta = dict(epoch=runner.epoch, iter=runner.iter)

        # handle auto in self.key_indicators and self.rules before the loop
        if "auto" in self.key_indicators:
            self._init_rule(self.rules, [list(metrics.keys())[0]])

        best_ckpt_updated = False
        # save best logic
        # get score from messagehub
        for key_indicator, rule in zip(self.key_indicators, self.rules):
            key_score = metrics[key_indicator]

            if len(self.key_indicators) == 1:
                best_score_key = "best_score"
                runtime_best_ckpt_key = "best_ckpt"
                best_ckpt_path = self.best_ckpt_path
            else:
                best_score_key = f"best_score_{key_indicator}"
                runtime_best_ckpt_key = f"best_ckpt_{key_indicator}"
                best_ckpt_path = self.best_ckpt_path_dict[key_indicator]

            if best_score_key not in runner.message_hub.runtime_info:
                best_score = self.init_value_map[rule]
            else:
                best_score = runner.message_hub.get_info(best_score_key)

            if key_score is None or not self.is_better_than[key_indicator](
                key_score, best_score
            ):
                continue

            best_ckpt_updated = True

            best_score = key_score
            runner.message_hub.update_info(best_score_key, best_score)

            if best_ckpt_path and is_main_process():
                is_removed = False
                if self.file_backend.isfile(best_ckpt_path):
                    self.file_backend.remove(best_ckpt_path)
                    is_removed = True
                elif self.file_backend.isdir(best_ckpt_path):
                    # checkpoints saved by deepspeed are directories
                    self.file_backend.rmtree(best_ckpt_path)
                    is_removed = True

                if is_removed:
                    runner.logger.info(
                        f"The previous best checkpoint {best_ckpt_path} " "is removed"
                    )

            best_ckpt_name = f"best_{key_indicator}_{ckpt_filename}"
            # Replace illegal characters for filename with `_`
            best_ckpt_name = best_ckpt_name.replace("/", "_")
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = self.file_backend.join_path(
                    self.out_dir, best_ckpt_name
                )
                runner.message_hub.update_info(
                    runtime_best_ckpt_key, self.best_ckpt_path
                )
            else:
                self.best_ckpt_path_dict[key_indicator] = self.file_backend.join_path(
                    self.out_dir, best_ckpt_name
                )
                runner.message_hub.update_info(
                    runtime_best_ckpt_key, self.best_ckpt_path_dict[key_indicator]
                )
            runner.save_checkpoint(
                self.out_dir,
                filename=best_ckpt_name,
                save_optimizer=False,
                save_param_scheduler=False,
                meta=meta,
                by_epoch=False,
                backend_args=self.backend_args,
            )
            runner.logger.info(
                f"The best checkpoint with {best_score:0.4f} {key_indicator} "
                f"at {cur_time} {cur_type} is saved to {best_ckpt_name}."
            )

        # save checkpoint again to update the best_score and best_ckpt stored
        # in message_hub because the checkpoint saved in `after_train_epoch`
        # or `after_train_iter` stage only keep the previous best checkpoint
        # not the current best checkpoint which causes the current best
        # checkpoint can not be removed when resuming training.
        if best_ckpt_updated and self.last_ckpt is not None:
            self._save_checkpoint_with_step(runner, cur_time, meta)

    def _init_rule(self, rules, key_indicators) -> None:
        """Initialize rule, key_indicator, comparison_func, and best score. If
        key_indicator is a list of string and rule is a string, all metric in
        the key_indicator will share the same rule.

        Here is the rule to determine which rule is used for key indicator when
        the rule is not specific (note that the key indicator matching is case-
        insensitive):

        1. If the key indicator is in ``self.greater_keys``, the rule
            will be specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule
            will be specified as 'less'.
        3. Or if any one item in ``self.greater_keys`` is a substring of
            key_indicator, the rule will be specified as 'greater'.
        4. Or if any one item in ``self.less_keys`` is a substring of
            key_indicator, the rule will be specified as 'less'.

        Args:
            rule (List[Optional[str]]): Comparison rule for best score.
            key_indicator (List[str]): Key indicator to determine
                the comparison rule.
        """
        if len(rules) == 1:
            rules = rules * len(key_indicators)

        self.rules = []
        for rule, key_indicator in zip(rules, key_indicators):
            if rule not in self.rule_map and rule is not None:
                raise KeyError(
                    "rule must be greater, less or None, " f"but got {rule}."
                )

            if rule is None and key_indicator != "auto":
                # `_lc` here means we use the lower case of keys for
                # case-insensitive matching
                key_indicator_lc = key_indicator.lower()
                greater_keys = {key.lower() for key in self.greater_keys}
                less_keys = {key.lower() for key in self.less_keys}

                if key_indicator_lc in greater_keys:
                    rule = "greater"
                elif key_indicator_lc in less_keys:
                    rule = "less"
                elif any(key in key_indicator_lc for key in greater_keys):
                    rule = "greater"
                elif any(key in key_indicator_lc for key in less_keys):
                    rule = "less"
                else:
                    raise ValueError(
                        "Cannot infer the rule for key "
                        f"{key_indicator}, thus a specific rule "
                        "must be specified."
                    )
            if rule is not None:
                self.is_better_than[key_indicator] = self.rule_map[rule]
            self.rules.append(rule)

        self.key_indicators = key_indicators

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs=Optional[dict],
    ) -> None:
        """Save the checkpoint and synchronize buffers after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_train_iters(runner, self.interval) or (
            self.save_last and self.is_last_train_iter(runner)
        ):
            runner.logger.info(f"Saving checkpoint at {runner.iter + 1} iterations")
            self._save_checkpoint(runner)
