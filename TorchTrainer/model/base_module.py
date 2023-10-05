import copy
import logging
from abc import ABCMeta
from typing import Iterable, List, Optional, Union
import torch.nn as nn

from TorchTrainer.utils.logging import print_log
from .weight_init import PretrainedInit, initialize
from .wrappers.utils import is_model_wrapper


class BaseModule(nn.Module, metaclass=ABCMeta):
    """模型构建的结构基础类, 继承自 `torch.nn.Module`,
        额外支持通过 `init_cfg` 进行结构参数选择初始化和自定义 print 操作.

        init_cfg 示例:
            # 指定属性名称(name)或结构类型(layer)初始化
            dict(type='Normal', layer='Conv2d', name='conv', mean=0, std=1, bias=0, ***),
            # 指定预训练模型地址进行初始化
            init_cfg = dict(type="Pretrained", checkpoint="torchvision://resnet50")
            # 指定由列表包含多个初始化字典选项, 会依次进行初始化
            init_cfg = [, ***]

    Args:
        init_cfg (dict or List[dict], optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__()
        self.is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    def init_weights(self):
        """根据 init_cfg 初始化模型的权重"""
        module_name = self.__class__.__name__
        if not self.is_init:
            if self.init_cfg:
                print_log(
                    f"initialize {module_name} with init_cfg {self.init_cfg}",
                    logger="current",
                    level=logging.DEBUG,
                )

                init_cfgs = self.init_cfg
                if isinstance(self.init_cfg, dict):
                    init_cfgs = [self.init_cfg]

                # 对多个初始化选项, 分离出预训练初始化选项
                other_cfgs = []
                pretrained_cfg = []
                for init_cfg in init_cfgs:
                    assert isinstance(init_cfg, dict)
                    if (
                        init_cfg["type"] == "Pretrained"
                        or init_cfg["type"] is PretrainedInit
                    ):
                        pretrained_cfg.append(init_cfg)
                    else:
                        other_cfgs.append(init_cfg)
                # 对当前层进行初始化
                initialize(self, other_cfgs)

            # 对子层进行初始化
            for m in self.children():
                if is_model_wrapper(m) and not hasattr(m, "init_weights"):
                    m = m.module
                if hasattr(m, "init_weights"):
                    m.init_weights()

            if self.init_cfg and pretrained_cfg:
                initialize(self, pretrained_cfg)
            self.is_init = True
        else:
            print_log(
                f"init_weights of {self.__class__.__name__} has "
                f"been called more than once.",
                logger="current",
                level=logging.WARNING,
            )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class Sequential(BaseModule, nn.Sequential):
    """Sequential 模块"""

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList 模块"""

    def __init__(
        self, modules: Optional[Iterable] = None, init_cfg: Optional[dict] = None
    ):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict 模块"""

    def __init__(self, modules: Optional[dict] = None, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)
