from typing import Any, Dict, Union

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from TorchTrainer.optim import OptimWrapper
from TorchTrainer.utils.registry import MODEL_WRAPPERS
from ..utils import detect_anomalous_params

MODEL_WRAPPERS.register_module(module=DistributedDataParallel)
MODEL_WRAPPERS.register_module(module=DataParallel)


@MODEL_WRAPPERS.register_module()
class TTDistributedDataParallel(DistributedDataParallel):
    """分布式模型包装器, 与 `DistributedDataParallel` 的区别在于:
    - 添加了 `train_step`, `val_step`, `test_step` 方法, 使其支持模型的前向计算过程.
        - `train_step`: 由 `runner.train_loop` 调用, 实现默认的模型前向计算, 梯度反向传播, 参数更新逻辑.
          为了利用 `DistributedDataParallel` 的自动梯度同步, `train_step` 调用 `DistributedDataParallel.forward`
          计算损失, 并调用 `BaseModel` 的其他方法进行数据预处理和解析损失. 最后, 通过 `OptimWrapper` 更新模型参数,
          并返回用于记录的损失字典.
        - `val_step`: 由 `runner.val_loop` 调用, 获取模型的预测结果. 由于没有梯度同步的要求, 该过程等价于 `BaseModel.val_step`
        - `test_step`: 由 `runner.test_loop` 调用, 等价于 `val_step`
    - 添加了 `detect_anomalous_params` 参数, 用于检测不在计算图中的异常参数(未使用或未参与损失计算), 仅用于调试, 会降低训练速度.
    - 支持多个子模型的优化器策略, 因此可以通过继承 `TTDistributedDataParallel`, 重写 `train_step` 方法, 实现自定义的优化器策略.
    """

    def __init__(self, module, detect_anomalous_params: bool = False, **kwargs):
        super().__init__(module=module, **kwargs)
        self.detect_anomalous_params = detect_anomalous_params

    def train_step(
        self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper
    ) -> Dict[str, torch.Tensor]:
        """模型训练过程中的前向、后向和参数更新接口. 
        - 调用 ``module.preprocess`` 进行数据预处理. 
        - 调用 ``module.forward(**data)`` 计算损失. 
        - 解析损失. 
        - 调用 ``optim_wrapper.optimizer_step`` 更新参数. 
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, training=True)
            losses = self._run_forward(data, mode="loss")
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        if self.detect_anomalous_params:
            detect_anomalous_params(parsed_loss, model=self)
        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """在验证过程中获取模块的预测结果. """
        return self.module.val_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """在测试过程中获取模块的预测结果. """
        return self.module.test_step(data)

    def _run_forward(self, data: Union[dict, tuple, list], mode: str) -> Any:
        """解包字典数据以供 :meth:`forward` 方法使用"""
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError(
                "Output of `data_preprocessor` should be "
                f"list, tuple or dict, but got {type(data)}"
            )
        return results
