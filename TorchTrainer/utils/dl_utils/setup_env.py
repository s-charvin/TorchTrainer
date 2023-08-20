import os
import platform
import warnings
import logging
import torch.multiprocessing as mp

from TorchTrainer.utils import digit_version
from TorchTrainer.utils.logging import print_log


def setup_cache_size_limit_of_dynamo():
    """Setup cache size limit of dynamo.

    Note: Due to the dynamic shape of the loss calculation and
    post-processing parts in the object detection algorithm, these
    functions must be compiled every time they are run.
    Setting a large value for torch._dynamo.config.cache_size_limit
    may result in repeated compilation, which can slow down training
    and testing speed. Therefore, we need to set the default value of
    cache_size_limit smaller. An empirical value is 4.
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
    """Set multi-processing related environment.

    Args:
        mp_start_method (str): Set the method which should be used to start
            child processes. Defaults to 'fork'.
        opencv_num_threads (int): Number of threads for opencv.
            Defaults to 0.
        distributed (bool): True if distributed environment.
            Defaults to False.
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
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
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
