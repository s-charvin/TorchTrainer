import logging
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from TorchTrainer.utils.logging import MessageHub, print_log
from TorchTrainer.utils.registry import OPTIM_WRAPPERS
from TorchTrainer.utils.dl_utils import has_batch_norm


@OPTIM_WRAPPERS.register_module()
class OptimWrapper:
    """OptimWrapper 为单精度训练和不同硬件的自动混合精度训练提供了用于更新参数的通用接口
    OptimWrapper 基于 torch.optim.Optimizer, 为常用的训练技巧(如梯度累积和梯度裁剪) 提供了简化的接口.
    子类只需要重写一些方法来实现混合精度训练.
        - 子类需要确保在调用 `update_params` 后, `self._inner_count += 1` 被自动执行.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        accumulative_counts: int = 1,
        clip_grad: Optional[dict] = None,
    ):
        assert (
            accumulative_counts > 0
        ), "_accumulative_counts at least greater than or equal to 1"
        self._accumulative_counts = accumulative_counts

        assert isinstance(optimizer, Optimizer), (
            "optimizer must be a `torch.optim.Optimizer` instance, but got "
            f"{type(optimizer)}"
        )
        self.optimizer = optimizer

        if clip_grad is not None:
            # clip_grad_kwargs should not be non-empty dict.
            assert isinstance(clip_grad, dict) and clip_grad, (
                "If `clip_grad` is not None, it should be a `dict` "
                "which is the arguments of `torch.nn.utils.clip_grad_norm_` "
                "or clip_grad_value_`."
            )
            clip_type = clip_grad.pop("type", "norm")
            if clip_type == "norm":
                self.clip_func = torch.nn.utils.clip_grad_norm_
                self.grad_name = "grad_norm"
            elif clip_type == "value":
                self.clip_func = torch.nn.utils.clip_grad_value_
                self.grad_name = "grad_value"
            else:
                raise ValueError(
                    'type of clip_grad should be "norm" or '
                    f'"value" but got {clip_type}'
                )
            assert clip_grad, (
                "`clip_grad` should contain other arguments "
                "besides `type`. The arguments should match "
                "with the `torch.nn.utils.clip_grad_norm_` or "
                "clip_grad_value_`"
            )
        self.clip_grad_kwargs = clip_grad
        self.message_hub = MessageHub.get_current_instance()
        self._inner_count = 0
        # `_max_counts` 表示参数更新的总次数. 它确保在 `_max_counts` 不能被 `accumulative_counts` 整除时, 最后几次迭代的梯度不会丢失. 
        self._max_counts = -1
        # `_remainder_iter` 用于计算最后几次迭代的损失因子. 如果 `_max_counts` 没有被初始化, 损失因子将始终与 `_accumulative_counts` 相同. 
        self._remainder_counts = -1

        # 以下代码用于初始化`base_param_settings`. `base_param_settings`用于存储不受优化器更新的参数. 在优化器中, 使用`base_param_settings`来跟踪基本学习率. 如果优化器有多个参数组, 则这些参数将不会按损失因子进行缩放. 
        if len(optimizer.param_groups) > 1:
            self.base_param_settings = {
                "params": torch.tensor([0.0], dtype=torch.float)
            }
            self.base_param_settings.update(**self.optimizer.defaults)
        else:
            self.base_param_settings = None

    def update_params(
        self,
        loss: torch.Tensor,
        step_kwargs: Optional[Dict] = None,
        zero_kwargs: Optional[Dict] = None,
    ) -> None:
        """Update parameters in :attr:`optimizer`."""
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)

        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """根据“_accumulative_counts”、“_inner_count”和 max_counts 获取缩放后的损失. 

        Args:
            loss (torch.Tensor): Original loss calculated by model.

        Returns:
            loss (torch.Tensor): Scaled loss.
        """
        if self._accumulative_counts == 1:
            # 不进行梯度累积的参数更新. 梯度不应该被重新缩放, `loss_factor=1`. 
            loss_factor = 1
        elif self._max_counts == -1:
            loss_factor = self._accumulative_counts
        else:
            # 如果 `self._accumulative_counts > 1`, 梯度需要重新缩放和累积. 
            # 在大多数情况下, `loss_factor` 等于 `self._accumulative_counts`. 
            # 然而, `self._max_counts` 可能不能被 `self._accumulative_counts` 整除, 
            # 所以最后几次迭代的 `loss_scale` 需要重新计算. 
            if self._inner_count < self._max_counts - self._remainder_counts:
                loss_factor = self._accumulative_counts
            else:
                loss_factor = self._remainder_counts
            assert loss_factor > 0, (
                "loss_factor should be larger than zero! This error could "
                "happened when initialize_iter_status called with an "
                "error `init_counts` or `max_counts`"
            )

        loss = loss / loss_factor
        return loss

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """执行梯度反向传播

        提供与自动混合精度训练兼容的统一“backward”接口. 子类可以重载此方法来实现所需的逻辑. 例如, “torch.cuda.amp”在反向过程中需要对GradScaler进行一些额外操作. 

        Note:
            如果子类继承自“OptimWrapper”, 则必须覆盖“backward”, 并实现“_inner_count +=1”. 

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`.
        """
        loss.backward(**kwargs)
        self._inner_count += 1

    def should_update(self) -> bool:
        """决定当前迭代是否应该更新参数
        # 只有当 `self._inner_count` 能被 `self._accumulative_counts` 整除, 或者 `self._inner_count` 等于 `self._max_counts` 时, 才更新参数. 
        """
        return (
            self._inner_count % self._accumulative_counts == 0
            or self._inner_count == self._max_counts
        )

    def step(self, **kwargs) -> None:
        """提供与自动混合精度训练兼容的统一“step”接口, 子类可以重载此方法来实现所需的逻辑. 例如, ``torch.cuda.amp``在步骤过程中需要对 ``GradScaler`` 进行一些额外操作. 
        如果 :attr:`clip_grad_kwargs` 不为 None, 则剪裁梯度, 然后更新参数. 
        """
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.step(**kwargs)

    def zero_grad(self, **kwargs) -> None:
        """提供与自动混合精度训练兼容的统一“zero_grad”接口"""
        self.optimizer.zero_grad(**kwargs)

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``.

        Provide unified ``state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be saved when training with ``torch.cuda.amp``.

        Returns:
            dict: The state dictionary of :attr:`optimizer`.
        """
        state_dict = self.optimizer.state_dict()
        if self.base_param_settings is not None:
            state_dict["base_param_settings"] = self.base_param_settings
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """提供与自动混合精度训练兼容的统一“load_state_dict”接口. 子类可以重载此方法来实现所需的逻辑. 例如, 在使用``torch.cuda.amp``进行训练时, 应加载GradScaler的状态字典. """
        base_param_settings = state_dict.pop("base_param_settings", None)

        if base_param_settings is not None:
            self.base_param_settings = base_param_settings

        # load state_dict of optimizer
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> List[dict]:
        """获取优化器的参数组,  为了兼容 :class:`_ParamScheduler`. """
        if self.base_param_settings is not None:
            return self.optimizer.param_groups + [self.base_param_settings]
        else:
            return self.optimizer.param_groups

    @property
    def defaults(self) -> dict:
        """获取优化器的默认参数,  为了兼容 :class:`_ParamScheduler`. """
        return self.optimizer.defaults

    def get_lr(self) -> Dict[str, List[float]]:
        """获取优化器的学习率. """
        res = {}
        if self.base_param_settings is not None:
            res["base_lr"] = [self.base_param_settings["lr"]]

        res["lr"] = [group["lr"] for group in self.optimizer.param_groups]

        return res

    def get_momentum(self) -> Dict[str, List[float]]:
        """获取优化器的动量 momentum. """
        momentum = []
        for group in self.optimizer.param_groups:
            # Get momentum of SGD.
            if "momentum" in group.keys():
                momentum.append(group["momentum"])
            # Get momentum of Adam.
            elif "betas" in group.keys():
                momentum.append(group["betas"][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """开启梯度累积和自动混合精度训练的上下文模式

        如果子类需要启用混合精度训练的上下文, 例如`AmpOptimWrapper`类, 则应在`optim_context`中启用相应的上下文. 
        由于`OptimWrapper`使用默认的 fp32 训练, 因此只有在梯度累积期间阻止不必要的梯度同步时才会启用 `optim_context`

        如果模型是具有“no_sync”方法 (即阻止梯度同步)且self._accumulative_counts != 1 的实例. 
        如果cur_iter可以被self._accumulative_counts整除, 则模型将不会自动同步梯度. 否则, 该方法将启用一个空上下文. 
        """
        # 在梯度累积过程中, 梯度同步应该只发生在更新参数之前. 
        if not self.should_sync() and hasattr(model, "no_sync"):
            with model.no_sync():
                yield
        else:
            yield

    def _clip_grad(self) -> None:
        """Clip the gradients of parameters."""
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group["params"])

        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad = self.clip_func(params, **self.clip_grad_kwargs)
            # `torch.nn.utils.clip_grad_value_` will return None.
            if grad is not None:
                self.message_hub.update_scalar(f"train/{self.grad_name}", float(grad))

    def initialize_count_status(
        self, model: nn.Module, init_counts: int, max_counts: int
    ) -> None:
        """初始化梯度累积所需的相关属性
        “OptimWrapper”可以在不调用“initialize_iter_status”的情况下使用. 
        然而, 考虑到“len(dataloader) == 10”, 且“accumulative_iter == 3”的情况. 
        由于10不能被3整除, 最后一次迭代将不会触发“optimizer.step()”, 会导致少一个参数更新. 
        """
        self._inner_count = init_counts
        self._max_counts = max_counts
        if self._inner_count % self._accumulative_counts != 0:
            print_log(
                "Resumed iteration number is not divisible by "
                "`_accumulative_counts` in `GradientCumulativeOptimizerHook`, "
                "which means the gradient of some iterations is lost and the "
                "result may be influenced slightly.",
                logger="current",
                level=logging.WARNING,
            )

        if has_batch_norm(model) and self._accumulative_counts > 1:
            print_log(
                "Gradient accumulative may slightly decrease "
                "performance because the model has BatchNorm layers.",
                logger="current",
                level=logging.WARNING,
            )
        # Remainder of `_max_counts` divided by `_accumulative_counts`
        self._remainder_counts = self._max_counts % self._accumulative_counts

    def should_sync(self) -> bool:
        """决定是否允许在当前迭代中进行自动梯度同步

        当使用梯度累积来跳过参数未更新的迭代时, 它会生效. 

        由于“should_sync”是由“optim_context”调用的, 并且在调用“backward”之前被调用, 这意味着“self._inner_count += 1”尚未发生. 
        因此, 在这里需要手动执行“self._inner_count += 1”. 
        """
        return (self._inner_count + 1) % self._accumulative_counts == 0 or (
            self._inner_count + 1
        ) == self._max_counts

    @property
    def inner_count(self):
        """获取优化器包装器的更新参数次数"""
        return self._inner_count

    def __repr__(self):
        wrapper_info = (
            f"Type: {type(self).__name__}\n"
            f"_accumulative_counts: {self._accumulative_counts}\n"
            "optimizer: \n"
        )
        optimizer_str = repr(self.optimizer) + "\n"
        return wrapper_info + optimizer_str
