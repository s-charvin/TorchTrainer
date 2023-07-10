from torchTraining.utils.dl_utils import TORCH_VERSION
from torchTraining.utils.version_utils import digit_version
from .averaged_model import (
    BaseAveragedModel,
    ExponentialMovingAverage,
    MomentumAnnealingEMA,
    StochasticWeightAverage,
)
from .utils import (
    convert_sync_batchnorm,
    detect_anomalous_params,
    merge_dict,
    revert_sync_batchnorm,
    stack_batch,
)
from .weight_init import (
    BaseInit,
    Caffe2XavierInit,
    ConstantInit,
    KaimingInit,
    NormalInit,
    PretrainedInit,
    TruncNormalInit,
    UniformInit,
    XavierInit,
    bias_init_with_prob,
    caffe2_xavier_init,
    constant_init,
    initialize,
    kaiming_init,
    normal_init,
    trunc_normal_init,
    uniform_init,
    xavier_init,
)
from .wrappers import (
    MMDistributedDataParallel,
    MMSeparateDistributedDataParallel,
    is_model_wrapper,
)

from .base_module import BaseModule, ModuleDict, ModuleList, Sequential
from .base_model import BaseModel
from .data_preprocessor import BaseDataPreprocessor, ImgDataPreprocessor


__all__ = [
    "MMDistributedDataParallel",
    "is_model_wrapper",
    "BaseAveragedModel",
    "StochasticWeightAverage",
    "ExponentialMovingAverage",
    "MomentumAnnealingEMA",
    "BaseModel",
    "BaseDataPreprocessor",
    "ImgDataPreprocessor",
    "MMSeparateDistributedDataParallel",
    "BaseModule",
    "stack_batch",
    "merge_dict",
    "detect_anomalous_params",
    "ModuleList",
    "ModuleDict",
    "Sequential",
    "revert_sync_batchnorm",
    "constant_init",
    "xavier_init",
    "normal_init",
    "trunc_normal_init",
    "uniform_init",
    "kaiming_init",
    "caffe2_xavier_init",
    "bias_init_with_prob",
    "BaseInit",
    "ConstantInit",
    "XavierInit",
    "NormalInit",
    "TruncNormalInit",
    "UniformInit",
    "KaimingInit",
    "Caffe2XavierInit",
    "PretrainedInit",
    "initialize",
    "convert_sync_batchnorm",
]

if digit_version(TORCH_VERSION) >= digit_version("1.11.0"):
    from .wrappers import MMFullyShardedDataParallel

    __all__.append("MMFullyShardedDataParallel")
