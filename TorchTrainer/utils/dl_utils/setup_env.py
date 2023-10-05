import os
import platform
import warnings
import logging
import torch.multiprocessing as mp

from TorchTrainer.utils import digit_version
from TorchTrainer.utils.logging import print_log


def setup_cache_size_limit_of_dynamo():
    """设置 Dynamo 的缓存大小限制.
    注意: 由于目标检测算法中损失计算和后处理部分的动态形状, 这些函数每次运行时都必须编译.
    设置 torch._dynamo.config.cache_size_limit 的较大值可能会导致重复编译, 这可能会降低训练和测试速度.
    因此, 我们需要将 cache_size_limit 的默认值设置得更小. 一个经验值是 4.
    """

    import torch

    if digit_version(torch.__version__) >= digit_version("2.0.0"):
        if "DYNAMO_CACHE_SIZE_LIMIT" in os.environ:
            import torch._dynamo

            cache_size_limit = int(os.environ["DYNAMO_CACHE_SIZE_LIMIT"])
            torch._dynamo.config.cache_size_limit = cache_size_limit
            print_log(
                f"torch._dynamo.config.cache_size_limit is force "
                f"set to {cache_size_limit}.",
                logger="current",
                level=logging.WARNING,
            )


def set_multi_processing(
    mp_start_method: str = "fork",
    opencv_num_threads: int = 0,
    distributed: bool = False,
) -> None:
    """设置多进程相关环境
    - 设置应该用于启动子进程的方法, 默认为'fork'.
    - 设置 opencv 多线程数为 0
    - 设置分布式环境(OMP_NUM_THREADS:1, MKL_NUM_THREADS:1)
    """
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != "Windows":
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f"Multi-processing start method `{mp_start_method}` is "
                f"different from the previous setting `{current_method}`."
                f"It will be force set to `{mp_start_method}`. You can "
                "change this behavior by changing `mp_start_method` in "
                "your config."
            )
        mp.set_start_method(mp_start_method, force=True)

    try:
        import cv2

        # disable opencv multithreading to avoid system being overloaded
        cv2.setNumThreads(opencv_num_threads)
    except ImportError:
        pass

    # setup OMP threads
    if "OMP_NUM_THREADS" not in os.environ and distributed:
        omp_num_threads = 1
        warnings.warn(
            "Setting OMP_NUM_THREADS environment variable for each process"
            f" to be {omp_num_threads} in default, to avoid your system "
            "being overloaded, please further tune the variable for "
            "optimal performance in your application as needed."
        )
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    # setup MKL threads
    if "MKL_NUM_THREADS" not in os.environ and distributed:
        mkl_num_threads = 1
        warnings.warn(
            "Setting MKL_NUM_THREADS environment variable for each process"
            f" to be {mkl_num_threads} in default, to avoid your system "
            "being overloaded, please further tune the variable for "
            "optimal performance in your application as needed."
        )
        os.environ["MKL_NUM_THREADS"] = str(mkl_num_threads)
