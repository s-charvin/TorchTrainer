import logging
from collections.abc import Callable
from typing import Generator, Optional, Type, TypeVar, Any, Dict, List, Tuple, Union
from rich.console import Console
from rich.table import Table

from TorchTrainer.utils import is_seq_of
from TorchTrainer.utils.logging import print_log



class Registry:
    """一个注册表类，用于注册模块，可以通过注册名称获取模块"""

    def __init__(
        self,
        name: str,
        build_func: Optional[Callable] = None,
        parent: Optional["Registry"] = None,
    ):
        from .build_functions import build_from_cfg

        self._name = name  # 注册名称
        self.module_dict: Dict[str, Type] = dict()  # 所包含的注册的模块
        self._children: Dict[str, "Registry"] = dict()  # 子注册
        self._imported = False  # 是否已经导入
        self.parent: Optional["Registry"] = None  # 父注册

        # 初始化父注册
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_child(self)
            self.parent = parent
        else:
            self.parent = None

        if build_func is None:
            if self.parent is not None:
                self.build_func = self.parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

    def _add_child(self, registry: "Registry") -> None:
        """Add a child for a registry.

        Args:
            registry (:obj:`Registry`): The ``registry`` will be added as a
                child of the ``self``.
        """

        assert isinstance(registry, Registry)
        assert (
            registry.name not in self.children
        ), f"{registry.name} exists in {self.name} registry"
        self.children[registry.name] = registry

    @property
    def name(self):
        return self._name

    @property
    def children(self):
        return self._children

    @property
    def root(self):
        return self._get_root_registry()

    def __len__(self):
        return len(self.module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f"Registry of {self._name}")
        table.add_column("Names", justify="left", style="cyan")
        table.add_column("Objects", justify="left", style="green")

        for name, obj in sorted(self.module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end="")

        return capture.get()

    def _get_root_registry(self) -> "Registry":
        """Return the root registry."""
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def get(self, key: str = None) -> Optional[Type]:
        """Get the registry record."""
        if key is None:
            return self.module_dict
        if not isinstance(key, str):
            raise TypeError(
                "The key argument of `Registry.get` must be a str, " f"got {type(key)}"
            )

        obj_self = None
        registry_name = self.name

        split_index = key.find(".")
        if split_index > 0:
            parent_Key, real_key = key[:split_index], key[split_index + 1 :]
        else:
            real_key = key
            parent_Key = None

        if parent_Key is None:
            # search in self -> parent -> children
            parent = self
            while parent is not None:
                if real_key in parent.module_dict:
                    obj_self = parent.module_dict[real_key]
                    registry_name = parent.name
                    break
                parent = parent.parent
            if obj_self is None:
                children = self._children
                while children is not None:
                    if real_key in children:
                        obj_self = children[real_key]
                        registry_name = obj_self.name
                        break
                    children = list(children.values())[0]._children
        else:
            root = self._get_root_registry()
            while root.name != parent_Key:
                root = root._children[parent_Key]
            split_index = real_key.find(".")
            while split_index != -1:
                parent_Key, real_key = (
                    real_key[:split_index],
                    real_key[split_index + 1 :],
                )
                root = root._children[parent_Key]
                split_index = real_key.find(".")
            if real_key in root.module_dict:
                obj_self = root.module_dict[real_key]
                registry_name = root.name

        if obj_self is not None:
            # For some rare cases (e.g. obj_self is a partial function), obj_self
            # doesn't have `__name__`. Use default value to prevent error
            self_name = getattr(obj_self, "__name__", str(obj_self))
            print_log(
                f'Get class/function `{self_name}` from "{registry_name}"',
                logger="current",
                level=logging.DEBUG,
            )
        return obj_self

    def __getitem__(self, key):
        return self.get(key)

    def build(self, cfg: dict, *args, **kwargs) -> Any:
        """Build an instance.

        Build an instance by calling :attr:`build_func`.
        """
        return self.build_func(cfg, registry=self, *args, **kwargs)

    def _register_module(
        self,
        module: Type,
        module_name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
    ) -> None:
        """Register a module."""
        if not callable(module):
            raise TypeError(f"module must be Callable, but got {type(module)}")
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self.module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(
                    f"{name} is already registered in {self.name} "
                    f"at {existed_module.__module__}"
                )
            self.module_dict[name] = module

    def register_module(
        self,
        name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
        module: Optional[Type] = None,
    ) -> Union[type, Callable]:
        """Register a module."""

        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be None, an instance of str, or a sequence of str, "
                f"but got {type(name)}"
            )

        # as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register
