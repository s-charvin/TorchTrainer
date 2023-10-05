import warnings
from math import inf, isfinite
from typing import Optional, Tuple, Union

from TorchTrainer.utils.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """当监测到指定指标达到平稳状态时, 提前停止训练. 

    参数:
        monitor (str): 用于决定提前停止的监测指标键. 
        rule (str, 可选): 比较规则. 可选值为'greater' (大于) 或'less' (小于) . 默认为None. 
        min_delta (float, 可选): 继续训练的最小差异. 默认为0.01. 
        strict (bool, 可选): 当`monitor`在`metrics`中找不到时, 是否中止训练. 默认为False. 
        check_finite: 当监测指标变为NaN或无穷大时, 是否停止训练. 默认为True. 
        patience (int, 可选): 在没有改进的情况下进行验证的次数, 超过该次数后将停止训练. 默认为5. 
        stopping_threshold (float, 可选): 一旦监测指标达到此阈值, 立即停止训练. 默认为None. 
    """

    priority = "LOWEST"

    rule_map = {"greater": lambda x, y: x > y, "less": lambda x, y: x < y}
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
        monitor: str,
        rule: Optional[str] = None,
        min_delta: float = 0.1,
        strict: bool = False,
        check_finite: bool = True,
        patience: int = 5,
        stopping_threshold: Optional[float] = None,
    ):
        self.monitor = monitor
        if rule is not None:
            if rule not in ["greater", "less"]:
                raise ValueError(
                    '`rule` should be either "greater" or "less", ' f"but got {rule}"
                )
        else:
            rule = self._init_rule(monitor)
        self.rule = rule
        self.min_delta = min_delta if rule == "greater" else -1 * min_delta
        self.strict = strict
        self.check_finite = check_finite
        self.patience = patience
        self.stopping_threshold = stopping_threshold

        self.wait_count = 0
        self.best_score = -inf if rule == "greater" else inf

    def _init_rule(self, monitor: str) -> str:
        greater_keys = {key.lower() for key in self._default_greater_keys}
        less_keys = {key.lower() for key in self._default_less_keys}
        monitor_lc = monitor.lower()
        if monitor_lc in greater_keys:
            rule = "greater"
        elif monitor_lc in less_keys:
            rule = "less"
        elif any(key in monitor_lc for key in greater_keys):
            rule = "greater"
        elif any(key in monitor_lc for key in less_keys):
            rule = "less"
        else:
            raise ValueError(
                f"Cannot infer the rule for {monitor}, thus rule " "must be specified."
            )
        return rule

    def _check_stop_condition(self, current_score: float) -> Tuple[bool, str]:
        compare = self.rule_map[self.rule]
        stop_training = False
        reason_message = ""

        if self.check_finite and not isfinite(current_score):
            stop_training = True
            reason_message = (
                f"Monitored metric {self.monitor} = "
                f"{current_score} is infinite. "
                f"Previous best value was "
                f"{self.best_score:.3f}."
            )

        elif self.stopping_threshold is not None and compare(
            current_score, self.stopping_threshold
        ):
            stop_training = True
            self.best_score = current_score
            reason_message = (
                f"Stopping threshold reached: "
                f"`{self.monitor}` = {current_score} is "
                f"{self.rule} than {self.stopping_threshold}."
            )
        elif compare(self.best_score + self.min_delta, current_score):
            self.wait_count += 1

            if self.wait_count >= self.patience:
                reason_message = (
                    f"the monitored metric did not improve "
                    f"in the last {self.wait_count} records. "
                    f"best score: {self.best_score:.3f}. "
                )
                stop_training = True
        else:
            self.best_score = current_score
            self.wait_count = 0

        return stop_training, reason_message

    def before_run(self, runner) -> None:
        """Check `stop_training` variable in `runner.train_loop`.

        Args:
            runner (Runner): The runner of the training process.
        """

        assert hasattr(
            runner.train_loop, "stop_training"
        ), "`train_loop` should contain `stop_training` variable."

    def after_val_epoch(self, runner, metrics):
        """Decide whether to stop the training process.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics
        """

        if self.monitor not in metrics:
            if self.strict:
                raise RuntimeError(
                    "Early stopping conditioned on metric "
                    f"`{self.monitor} is not available. Please check available"
                    f" metrics {metrics}, or set `strict=False` in "
                    "`EarlyStoppingHook`."
                )
            warnings.warn(
                "Skip early stopping process since the evaluation "
                f"results ({metrics.keys()}) do not include `monitor` "
                f"({self.monitor})."
            )
            return

        current_score = metrics[self.monitor]

        stop_training, message = self._check_stop_condition(current_score)
        if stop_training:
            runner.train_loop.stop_training = True
            runner.logger.info(message)
