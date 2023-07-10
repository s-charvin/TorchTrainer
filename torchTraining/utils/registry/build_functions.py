import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional, Union
from torchTraining.utils.registry import Registry
from torchTraining.config import Config, ConfigDict

if TYPE_CHECKING:
    import torch.nn as nn

    from torchTraining.optim.scheduler import _ParamScheduler
    from torchTraining.runner import Runner


def build_from_cfg(
    cfg: Union[dict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, Config]] = None,
) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.
    """

    from torchTraining.utils.logging import print_log

    if not isinstance(cfg, (dict, Config)):
        raise TypeError(f"cfg should be a dict, Config, but got {type(cfg)}")
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be a torchTraining.Registry object, "
            f"but got {type(registry)}"
        )
    if not (isinstance(default_args, (dict, Config)) or default_args is None):
        raise TypeError(
            "default_args should be a dict, Config or None, "
            f"but got {type(default_args)}"
        )

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f"{obj_type} is not in the {registry.name} registry. "
                f"Please check whether the value of `{obj_type}` is "
                "correct or it was registered as expected. "
            )
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

    try:
        obj = obj_cls(**args)
        if (
            inspect.isclass(obj_cls)
            or inspect.isfunction(obj_cls)
            or inspect.ismethod(obj_cls)
        ):
            print_log(
                f"An `{obj_cls.__name__}` instance is built from "
                "registry, and its implementation can be found in "
                f"{obj_cls.__module__}",
                logger="current",
                level=logging.DEBUG,
            )
        else:
            print_log(
                "An instance is built from registry, and its constructor "
                f"is {obj_cls}",
                logger="current",
                level=logging.DEBUG,
            )
        return obj

    except Exception as e:
        cls_location = "/".join(obj_cls.__module__.split("."))
        raise type(e)(
            f"class `{obj_cls.__name__}` in " f"{cls_location}.py: {e}"
        )


def build_runner_from_cfg(
    cfg: Union[dict, ConfigDict, Config], registry: Registry
) -> "Runner":
    """Build a Runner object.

    Examples:
        >>> from torchTraining.registry import Registry, build_runner_from_cfg
        >>> RUNNERS = Registry('runners', build_func=build_runner_from_cfg)
        >>> @RUNNERS.register_module()
        >>> class CustomRunner(Runner):
        >>>     def setup_env(env_cfg):
        >>>         pass
        >>> cfg = dict(runner_type='CustomRunner', ...)
        >>> custom_runner = RUNNERS.build(cfg)

    Args:
        cfg (dict or ConfigDict or Config): Config dict. If "runner_type" key
            exists, it will be used to build a custom runner. Otherwise, it
            will be used to build a default runner.
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        object: The constructed runner object.
    """
    from torchTraining.utils.logging import print_log

    assert isinstance(
        cfg, (dict, ConfigDict, Config)
    ), f"cfg should be a dict, ConfigDict or Config, but got {type(cfg)}"
    assert isinstance(registry, Registry), (
        "registry should be a torchTraining.Registry object",
        f"but got {type(registry)}",
    )

    args = cfg.copy()

    obj_type = args.get("runner_type", "Runner")
    if isinstance(obj_type, str):
        runner_cls = registry.get(obj_type)
        if runner_cls is None:
            raise KeyError(
                f"{obj_type} is not in the {registry.name} registry. "
                f"Please check whether the value of `{obj_type}` is "
                "correct or it was registered as expected. More details "
                "can be found at https://torchTraining.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module"
            )
    elif inspect.isclass(obj_type):
        runner_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

    try:
        runner = runner_cls.from_cfg(args)
        print_log(
            f"An `{runner_cls.__name__}` instance is built from "
            "registry, its implementation can be found in"
            f"{runner_cls.__module__}",
            logger="current",
            level=logging.DEBUG,
        )
        return runner

    except Exception as e:
        # Normal TypeError does not print class name.
        cls_location = "/".join(runner_cls.__module__.split("."))
        raise type(e)(
            f"class `{runner_cls.__name__}` in "
            f"{cls_location}.py: {e}"
        )


def build_model_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, "ConfigDict", "Config"]] = None,
) -> "nn.Module":
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, which is either a config
            dict or a list of config dicts. If cfg is a list, the built
            modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn.Module.
    """
    from torchTraining.model import Sequential

    if isinstance(cfg, list):
        modules = [build_from_cfg(_cfg, registry, default_args) for _cfg in cfg]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_scheduler_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, ConfigDict, Config]] = None,
) -> "_ParamScheduler":
    """Builds a ``ParamScheduler`` instance from config.

    ``ParamScheduler`` supports building instance by its constructor or
    method ``build_iter_from_epoch``. Therefore, its registry needs a build
    function to handle both cases.

    Args:
        cfg (dict or ConfigDict or Config): Config dictionary. If it contains
            the key ``convert_to_iter_based``, instance will be built by method
            ``convert_to_iter_based``, otherwise instance will be built by its
            constructor.
        registry (:obj:`Registry`): The ``PARAM_SCHEDULERS`` registry.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. It must contain key ``optimizer``. If
            ``convert_to_iter_based`` is defined in ``cfg``, it must
            additionally contain key ``epoch_length``. Defaults to None.

    Returns:
        object: The constructed ``ParamScheduler``.
    """
    assert isinstance(
        cfg, (dict, ConfigDict, Config)
    ), f"cfg should be a dict, ConfigDict or Config, but got {type(cfg)}"
    assert isinstance(registry, Registry), (
        "registry should be a torchTraining.Registry object",
        f"but got {type(registry)}",
    )

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    convert_to_iter = args.pop("convert_to_iter_based", False)
    if convert_to_iter:
        scheduler_type = args.pop("type")
        assert "epoch_length" in args and args.get("by_epoch", True), (
            "Only epoch-based parameter scheduler can be converted to "
            "iter-based, and `epoch_length` should be set"
        )
        if isinstance(scheduler_type, str):
            scheduler_cls = registry.get(scheduler_type)
            if scheduler_cls is None:
                raise KeyError(
                    f"{scheduler_type} is not in the {registry.name} "
                    "registry. Please check whether the value of "
                    f"`{scheduler_type}` is correct or it was registered "
                    "as expected. More details can be found at https://torchTraining.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module"
                )
        elif inspect.isclass(scheduler_type):
            scheduler_cls = scheduler_type
        else:
            raise TypeError(
                "type must be a str or valid type, but got " f"{type(scheduler_type)}"
            )
        return scheduler_cls.build_iter_from_epoch(**args)
    else:
        args.pop("epoch_length", None)
        return build_from_cfg(args, registry)
