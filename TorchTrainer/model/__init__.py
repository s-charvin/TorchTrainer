from TorchTrainer.utils.dl_utils import TORCH_VERSION
from TorchTrainer.utils.version_utils import digit_version

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
    TTDistributedDataParallel,
    TTSeparateDistributedDataParallel,
    is_model_wrapper,
)

from .base_module import BaseModule, ModuleDict, ModuleList, Sequential
from .base_model import BaseModel
from .data_preprocessor import BaseDataPreprocessor, ImgDataPreprocessor


__all__ = [
    "BaseModule",
    "BaseModel",
    "TTDistributedDataParallel",
    "TTSeparateDistributedDataParallel",
    "is_model_wrapper",
    "BaseDataPreprocessor",
    "ImgDataPreprocessor",
    "stack_batch",
    "merge_dict",
    "detect_anomalous_params",
    "revert_sync_batchnorm",
    "convert_sync_batchnorm",
    "ModuleList",
    "ModuleDict",
    "Sequential",
    # 参数初始化
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
]
