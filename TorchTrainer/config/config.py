import ast
import copy
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple, Union
from collections import OrderedDict, abc
from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

from TorchTrainer.utils.fileio import dump, load
from TorchTrainer.utils.logging import print_log
from TorchTrainer.utils import (
    check_file_exist,
)

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
RESERVED_KEYS = ["filename", "text", "pretty_text", "env_variables"]

if platform.system() == "Windows":
    import regex as re
else:
    import re


class RemoveAssignFromAST(ast.NodeTransformer):
    """从抽象语法树(AST)中删除指定名称的赋值语句(Assign node). """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == self.key:
            return None
        else:
            return node


class ConfigDict(Dict):
    """自定义字典类型, 支持 python 字典操作和 类似 `dict.key` 的属性访问方式."""

    def __init__(__self, *args, **kwargs):
        for arg in args:
            if not arg:
                continue
            if isinstance(arg, ConfigDict):
                for key, val in dict.items(arg):
                    __self[key] = __self._hook(val)
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in dict.items(kwargs):
            __self[key] = __self._hook(val)

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )
        except Exception as e:
            raise e
        else:
            return value

    @classmethod
    def _hook(cls, item):
        # avoid to convert user defined dict to ConfigDict.
        if type(item) in (dict, OrderedDict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __setattr__(self, name, value):
        value = self._hook(value)
        return super().__setattr__(name, value)

    def __setitem__(self, name, value):
        value = self._hook(value)
        return super().__setitem__(name, value)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in super().items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs) -> None:
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError("update only accept one positional argument")
            for key, value in dict.items(args[0]):
                other[key] = value
        for key, value in dict(kwargs).items():
            other[key] = value
        for k, v in other.items():
            if (
                (k not in self)
                or (not isinstance(self[k], dict))
                or (not isinstance(v, dict))
            ):
                self[k] = self._hook(v)
            else:
                self[k].update(v)


class Config:
    """配置参数管理类
    - 通过 `Config.fromfile` 函数解析 `python`、`json` 和 `yaml` 类型文件的配置内容.
    - 通过 `Config.fromstring` 函数解析配置字符串, 需指定对应内容类型.
    - 支持类似 `Config.key` 的属性访问操作.
    - 支持配置文件中使用 `{{var}}` 的方式使用预定义变量, 如 `fileDirname`, `fileBasename`, `fileBasenameNoExtension`, `fileExtname`.
    - 支持配置文件中使用 `{{var:default}}` 的方式使用环境变量, 其中 `default` 为环境变量不存在时的默认值.
    - 支持通过 `_base_` 变量实现配置继承, 实现多个配置文件合并功能.
    """

    def __init__(
        self,
        cfg_dict: dict = None,
        cfg_text: Optional[str] = None,
        filename: Optional[Union[str, Path]] = None,
        env_variables: Optional[dict] = None,
    ):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")
        # 因为重写了当前类的 __setattr__ 方法, 所以使用父类 object 的 __setattr__ 方法来添加属性, 避免在当前类出现递归调用
        super().__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super().__setattr__("filename", filename)

        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""
        super().__setattr__("text", text)
        if env_variables is None:
            env_variables = dict()
        super().__setattr__("_env_variables", env_variables)

    @staticmethod
    def fromfile(
        filename: Union[str, Path],
        use_predefined_variables: bool = True,
        use_environment_variables: bool = True,
    ) -> "Config":
        """从配置文件中加载配置参数."""
        filename = str(filename) if isinstance(filename, Path) else filename
        cfg_dict, cfg_text, env_variables = Config._file2dict(
            filename, use_predefined_variables, use_environment_variables
        )
        return Config(
            cfg_dict, cfg_text=cfg_text, filename=filename, env_variables=env_variables
        )

    @staticmethod
    def fromstring(cfg_str: str, file_format: str) -> "Config":
        """从配置字符串中加载配置参数."""
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn('Please check "file_format", the file format may be .py')

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=file_format, delete=False
        ) as temp_file:
            temp_file.write(cfg_str)
        for i in [
            "{{fileDirname}}",
            "{{fileBasename}}",
            "{{fileBasenameNoExtension}}",
            "{{fileExtname}}",
        ]:
            assert (
                i not in cfg_str
            ), "When using the fromstring function to load configuration parameters, the use of basic variables is not supported, because it will be stored in a temporary file, making the basic variables invalid."
        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)  # manually delete the temporary file
        return cfg

    @staticmethod
    def _validate_py_syntax(filename: str):
        """验证 python 配置文件的语法是否正确."""
        with open(filename, encoding="utf-8") as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(
                "There are syntax errors in config " f"file {filename}: {e}"
            )

    @staticmethod
    def _substitute_predefined_vars(filename: str, temp_config_name: str):
        """替换配置文件中的预定义变量."""
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname,
        )
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            value = value.replace("\\", "/")
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _substitute_env_variables(filename: str, temp_config_name: str):
        """替换配置文件中的环境变量."""
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        regexp = r"\{\{[\'\"]?\s*\$(\w+)\s*\:\s*(\S*?)\s*[\'\"]?\}\}"
        env_variables = dict()
        while True:
            match = re.search(regexp, config_file)
            if match:
                var_name = match.group(1)
                value = match.group(2)
                if var_name in os.environ:
                    value = os.environ[var_name].replace("\\", "/")
                    env_variables[var_name] = value
                    print_log(
                        f"Using env variable `{var_name}` with value of "
                        f"{value} to replace item in config.",
                        logger="current",
                    )
                if not value:
                    raise KeyError(
                        f"`{var_name}` cannot be found in `os.environ`."
                        f" Please set `{var_name}` in environment or "
                        "give a default value."
                    )
                config_file = config_file.replace(match.group(0), value)
            else:
                break

        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return env_variables

    @staticmethod
    def _pre_substitute_base_vars(filename: str, temp_config_name: str) -> dict:
        """替换配置文件中的 base 变量之前的预处理."""
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        base_var_dict = {}
        regexp = r"\{\{\s*" + BASE_KEY + r"\.([\w\.]+)\s*\}\}"
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f"_{base_var}_{uuid.uuid4().hex.lower()[:6]}"
            base_var_dict[randstr] = base_var
            regexp = r"\{\{\s*" + BASE_KEY + r"\." + base_var + r"\s*\}\}"
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg: Any, base_var_dict: dict, base_cfg: dict) -> Any:
        """替换配置文件中的 base 变量."""
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split("."):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            )
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split("."):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _file2dict(
        filename: str,
        use_predefined_variables: bool = True,
        use_environment_variables: bool = True,
    ) -> Tuple[dict, str, dict]:
        """实现配置文件到字典的转换."""
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname
            )
            if platform.system() == "Windows":
                temp_config_file.close()

            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename, temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            # Substitute environment variables
            env_variables = dict()
            if use_environment_variables:
                env_variables = Config._substitute_env_variables(
                    temp_config_file.name, temp_config_file.name
                )
            # Substitute base variables from placeholders to strings
            base_var_dict = Config._pre_substitute_base_vars(
                temp_config_file.name, temp_config_file.name
            )

            # Handle base files
            base_cfg_dict = ConfigDict()
            cfg_text_list = list()
            for base_cfg_path in Config._get_base_files(temp_config_file.name):
                base_cfg_path = Config._get_cfg_path(base_cfg_path, filename)
                _cfg_dict, _cfg_text, _env_variables = Config._file2dict(
                    filename=base_cfg_path,
                    use_predefined_variables=use_predefined_variables,
                    use_environment_variables=use_environment_variables,
                )
                cfg_text_list.append(_cfg_text)
                env_variables.update(_env_variables)
                duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError(
                        "Duplicate key is not allowed among bases. "
                        f"Duplicate keys: {duplicate_keys}"
                    )

                _cfg_dict = Config._dict_to_config_dict(_cfg_dict)
                base_cfg_dict.update(_cfg_dict)

            # Handle current ".py" config file
            if filename.endswith(".py"):
                with open(temp_config_file.name, encoding="utf-8") as f:
                    codes = ast.parse(f.read())
                    codes = RemoveAssignFromAST(BASE_KEY).visit(codes)
                codeobj = compile(codes, "", mode="exec")
                # Support load global variable in nested function of the config.
                global_locals_var = {"_base_": base_cfg_dict}
                ori_keys = set(global_locals_var.keys())
                eval(codeobj, global_locals_var, global_locals_var)
                cfg_dict = {
                    key: value
                    for key, value in global_locals_var.items()
                    if (key not in ori_keys and not key.startswith("__"))
                }
            # Handle current ".json" or ".yaml" config file
            elif filename.endswith((".yml", ".yaml", ".json")):
                cfg_dict = load(temp_config_file.name)

            # close temp file
            for key, value in list(cfg_dict.items()):
                if isinstance(value, (types.FunctionType, types.ModuleType)):
                    cfg_dict.pop(key)
            temp_config_file.close()

        cfg_text = filename + "\n"
        with open(filename, encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        # Substitute base variables from strings to their actual values
        cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict, base_cfg_dict)
        cfg_dict.pop(BASE_KEY, None)

        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith("__")}

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text, env_variables

    @staticmethod
    def _dict_to_config_dict(cfg: dict):
        """递归地将所有字典转换为 ConfigDict 类型."""
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
            for key, value in cfg.items():
                cfg[key] = Config._dict_to_config_dict(value)
        elif isinstance(cfg, tuple):
            cfg = tuple(Config._dict_to_config_dict(_cfg) for _cfg in cfg)
        elif isinstance(cfg, list):
            cfg = [Config._dict_to_config_dict(_cfg) for _cfg in cfg]
        return cfg

    @staticmethod
    def _get_base_files(filename: str) -> list:
        """获取配置文件中的 base 文件."""
        file_format = osp.splitext(filename)[1]
        if file_format == ".py":
            Config._validate_py_syntax(filename)
            with open(filename, encoding="utf-8") as f:
                codes = ast.parse(f.read()).body

                def is_base_line(c):
                    return (
                        isinstance(c, ast.Assign)
                        and isinstance(c.targets[0], ast.Name)
                        and c.targets[0].id == BASE_KEY
                    )

                base_code = next((c for c in codes if is_base_line(c)), None)
                if base_code is not None:
                    base_code = ast.Expression(body=base_code.value)
                    base_files = eval(compile(base_code, "", mode="eval"))
                else:
                    base_files = []
        elif file_format in (".yml", ".yaml", ".json"):
            import TorchTrainer

            cfg_dict = TorchTrainer.load(filename)
            base_files = cfg_dict.get(BASE_KEY, [])
        else:
            raise TypeError(
                "The config type should be py, json, yaml or "
                f"yml, but got {file_format}"
            )
        base_files = base_files if isinstance(base_files, list) else [base_files]
        return base_files

    @staticmethod
    def _get_cfg_path(cfg_path: str, filename: str) -> str:
        """获取配置文件的路径."""

        # Get local config path.
        cfg_dir = osp.dirname(filename)
        cfg_path = osp.join(cfg_dir, cfg_path)
        return cfg_path

    @staticmethod
    def _merge_a_into_b(a: dict, b: dict, allow_list_keys: bool = False) -> dict:
        """合并两个字典(非就地)."""
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types: Union[Tuple, type] = (
                        (dict, list) if allow_list_keys else dict
                    )
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f"{k}={v} in child config cannot inherit from "
                            f"base because {k} is a dict in the child config "
                            f"but is of type {type(b[k])} in base config. "
                            f"You may set `{DELETE_KEY}=True` to ignore the "
                            f"base config."
                        )
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b

    @property
    def pretty_text(self) -> str:
        """用于输出字符串类型的配置文件内容(Python代码类型)."""

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(
                    f"dict({_indent(_format_dict(v_), indent)})," for v_ in v
                ).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{str(k)}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style="pep8",
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
        )
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self) -> Tuple[dict, Optional[str], Optional[str], dict]:
        return (self._cfg_dict, self.filename, self.text, self._env_variables)

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __setstate__(self, state: Tuple[dict, Optional[str], Optional[str], dict]):
        _cfg_dict, _filename, _text, _env_variables = state

        super().__setattr__("_cfg_dict", _cfg_dict)
        super().__setattr__("filename", _filename)
        super().__setattr__("text", _text)
        super().__setattr__("_env_variables", _env_variables)

    def dump(self, file: Optional[Union[str, Path]] = None):
        """存储配置文件."""
        file = str(file) if isinstance(file, Path) else file
        cfg_dict = super().__getattribute__("_cfg_dict").to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith(".py"):
                return self.pretty_text
            else:
                file_format = self.filename.split(".")[-1]
                return dump(cfg_dict, file_format=file_format)
        elif file.endswith(".py"):
            with open(file, "w", encoding="utf-8") as f:
                f.write(self.pretty_text)
        else:
            file_format = file.split(".")[-1]
            return dump(cfg_dict, file=file, file_format=file_format)

    def merge_from_dict(self, options: dict, allow_list_keys: bool = True) -> None:
        """将字典类型的配置参数合并到当前配置参数中.
        - 支持 key 使用 "." 分割的方式自动转换为多层嵌套字典类型.
        - 支持将同层级数字类型的 key 转换为列表类型.
        """
        option_cfg_dict: dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v
        cfg_dict = super().__getattribute__("_cfg_dict")
        super().__setattr__(
            "_cfg_dict",
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys
            ),
        )
