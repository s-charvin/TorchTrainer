"""这里提供了统一的文件I/O相关函数, 支持根据指定的文件路径或 backend_args 操作不同的文件后端的I/O.

目前支持五种文件后端:
    - LocalBackend
    - PetrelBackend
    - HTTPBackend
    - LmdbBackend
    - MemcacheBackend

注意: 该模块提供了上述所有文件后端的并集, 因此如果文件后端中的接口没有实现, 将会抛出 NotImplementedError.
这里提供了两种调用文件后端方法的方式:
    - 通过 ``get_file_backend`` 初始化一个文件后端, 然后调用其方法.
    - 直接调用统一的I/O函数, 该函数会先调用 ``get_file_backend`` 获取对应的后端, 然后再调用对应的后端方法.
"""
import json
import warnings
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union

from TorchTrainer.utils import is_filepath, is_str
from .backends import backends, prefix_to_backends

# file_handlers and register_handler had been moved to
# TorchTrainer/fileio/handlers/registry_utis. Import them
# in this file to keep backward compatibility.
from .handlers import file_handlers, register_handler

backend_instances: dict = {}


def _parse_uri_prefix(uri: Union[str, Path]) -> str:
    """解析 uri 文件的前缀.

    Returns:
        str: Return the prefix of uri if the uri contains '://'. Otherwise,
        return ''.
    """
    assert is_filepath(uri)
    uri = str(uri)
    # if uri does not contains '://', the uri will be handled by
    # LocalBackend by default
    if "://" not in uri:
        return ""
    else:
        prefix, _ = uri.split("://")
        # In the case of PetrelBackend, the prefix may contain the cluster
        # name like clusterName:s3://path/of/your/file
        if ":" in prefix:
            _, prefix = prefix.split(":")
        return prefix


def _get_file_backend(prefix: str, backend_args: dict):
    """基于 uri 的前缀或 backend_args 返回对应的文件后端."""
    # backend name has a higher priority
    if "backend" in backend_args:
        # backend_args should not be modified
        backend_args_bak = backend_args.copy()
        backend_name = backend_args_bak.pop("backend")
        backend = backends[backend_name](**backend_args_bak)
    else:
        backend = prefix_to_backends[prefix](**backend_args)
    return backend


def get_file_backend(
    uri: Union[str, Path, None] = None,
    *,
    backend_args: Optional[dict] = None,
    enable_singleton: bool = False,
):
    """基于 uri 的前缀或 backend_args 返回对应的文件后端.
    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        enable_singleton (bool): Whether to enable the singleton pattern.
            If it is True, the backend created will be reused if the
            signature is same with the previous one. Defaults to False.
    """
    global backend_instances

    if backend_args is None:
        backend_args = {}

    if uri is None and "backend" not in backend_args:
        raise ValueError(
            'uri should not be None when "backend" does not exist in ' "backend_args"
        )

    if uri is not None:
        prefix = _parse_uri_prefix(uri)
    else:
        prefix = ""

    if enable_singleton:
        # TODO: whether to pass sort_key to json.dumps
        unique_key = f"{prefix}:{json.dumps(backend_args)}"
        if unique_key in backend_instances:
            return backend_instances[unique_key]

        backend = _get_file_backend(prefix, backend_args)
        backend_instances[unique_key] = backend
        return backend
    else:
        backend = _get_file_backend(prefix, backend_args)
        return backend


def get(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bytes:
    """从给定的 ``filepath`` 中读取字节数据.

    Args:
        filepath (str or Path): Path to read data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bytes: Expected bytes object.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    return backend.get(filepath)


def get_text(
    filepath: Union[str, Path],
    encoding="utf-8",
    backend_args: Optional[dict] = None,
) -> str:
    """从给定的 ``filepath`` 中读取文本数据.
    Args:
        filepath (str or Path): Path to read data.
        encoding (str): The encoding format used to open the ``filepath``.
            Defaults to 'utf-8'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: Expected text reading from ``filepath``.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    return backend.get_text(filepath, encoding)


def put(
    obj: bytes,
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> None:
    """向给定的 ``filepath`` 中写入字节数据.
    注意: ``put`` 应该在 ``filepath`` 的目录不存在时创建目录.

    Args:
        obj (bytes): Data to be written.
        filepath (str or Path): Path to write data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    backend.put(obj, filepath)


def put_text(
    obj: str,
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> None:
    """向给定的 ``filepath`` 中写入文本数据.
    注意: ``put_text`` 应该在 ``filepath`` 的目录不存在时创建目录.

    Args:
        obj (str): Data to be written.
        filepath (str or Path): Path to write data.
        encoding (str, optional): The encoding format used to open the
            ``filepath``. Defaults to 'utf-8'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    backend.put_text(obj, filepath)


def exists(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """检查 ``filepath`` 是否存在.

    Args:
        filepath (str or Path): Path to be checked whether exists.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    return backend.exists(filepath)


def isdir(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """检查 ``filepath`` 是否是一个目录.

    Args:
        filepath (str or Path): Path to be checked whether it is a
            directory.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    return backend.isdir(filepath)


def isfile(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """检查 ``filepath`` 是否是一个文件.

    Args:
        filepath (str or Path): Path to be checked whether it is a file.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    return backend.isfile(filepath)


def join_path(
    filepath: Union[str, Path],
    *filepaths: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    r"""拼接所有的文件路径.

    Args:
        filepath (str or Path): Path to be concatenated.
        *filepaths (str or Path): Other paths to be concatenated.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The result of concatenation.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    return backend.join_path(filepath, *filepaths)


@contextmanager
def get_local_path(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Generator[Union[str, Path], None, None]:
    """从 ``filepath`` 中下载数据并写入本地路径.
    此函数被 :meth:`contxtlib.contextmanager` 装饰, 可以使用 ``with`` 语句调用,
    当从 ``with`` 语句中退出时, 临时路径将被释放.

    注意: 如果 ``filepath`` 是一个本地路径, 则直接返回 ``filepath``. 否则, 会从
    ``filepath`` 中下载数据并写入本地路径, 然后返回本地路径.

    Args:
        filepath (str or Path): Path to be read data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Yields:
        Iterable[str]: Only yield one path.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    with backend.get_local_path(str(filepath)) as local_path:
        yield local_path


def copyfile(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """复制 ``src`` 文件到 ``dst`` 并返回 ``dst``.
    ``src`` 和 ``dst`` 应该有相同的前缀. 如果 ``dst`` 指定了一个目录, 则 ``src``
    将被复制到 ``dst`` 中, 使用 ``src`` 的基本文件名. 如果 ``dst`` 指定的文件已经
    存在, 它将被替换.

    Args:
        src (str or Path): A file to be copied.
        dst (str or Path): Copy file to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination file.

    Raises:
        SameFileError: If src and dst are the same file, a SameFileError will
            be raised.
    """
    backend = get_file_backend(src, backend_args=backend_args, enable_singleton=True)
    return backend.copyfile(src, dst)


def copytree(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """递归的复制 ``src`` 目录到 ``dst`` 并返回 ``dst``.
    ``src`` 和 ``dst`` 应该有相同的前缀. ``dst`` 必须不存在.

    Args:
        src (str or Path): A directory to be copied.
        dst (str or Path): Copy directory to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination directory.

    Raises:
        FileExistsError: If dst had already existed, a FileExistsError will be
            raised.
    """
    backend = get_file_backend(src, backend_args=backend_args, enable_singleton=True)
    return backend.copytree(src, dst)


def copyfile_from_local(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """从本地复制 ``src`` 文件到 ``dst`` 并返回 ``dst``.
    如果 backend 是 LocalBackend 的实例, 则与 :func:`copyfile` 做相同的事情.

    Args:
        src (str or Path): A local file to be copied.
        dst (str or Path): Copy file to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: If dst specifies a directory, the file will be copied into dst
        using the base filename from src.
    """
    backend = get_file_backend(dst, backend_args=backend_args, enable_singleton=True)
    return backend.copyfile_from_local(src, dst)


def copytree_from_local(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """递归的从本地复制 ``src`` 目录到 ``dst`` 并返回 ``dst``.
    如果 backend 是 LocalBackend 的实例, 则与 :func:`copytree` 做相同的事情.

    Args:
        src (str or Path): A local directory to be copied.
        dst (str or Path): Copy directory to dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination directory.
    """
    backend = get_file_backend(dst, backend_args=backend_args, enable_singleton=True)
    return backend.copytree_from_local(src, dst)


def copyfile_to_local(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """复制 ``src`` 文件到本地 ``dst`` 并返回 ``dst``.
    如果 backend 是 LocalBackend 的实例, 则与 :func:`copyfile` 做相同的事情.

    如果 ``dst`` 指定了一个目录, 则 ``src`` 将被复制到 ``dst`` 中, 使用 ``src`` 的
    基本文件名. 如果 ``dst`` 指定的文件已经存在, 它将被替换.

    Args:
        src (str or Path): A file to be copied.
        dst (str or Path): Copy file to to local dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: If dst specifies a directory, the file will be copied into dst
        using the base filename from src.
    """
    backend = get_file_backend(dst, backend_args=backend_args, enable_singleton=True)
    return backend.copyfile_to_local(src, dst)


def copytree_to_local(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """递归地将以 ``src`` 为根的整个目录树复制到名为 ``dst`` 的本地目录, 并返回 ``dst``.
    如果 backend 是 LocalBackend 的实例, 则与 :func:`copytree` 做相同的事情.
    Args:
        src (str or Path): A directory to be copied.
        dst (str or Path): Copy directory to local dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The destination directory.
    """
    backend = get_file_backend(dst, backend_args=backend_args, enable_singleton=True)
    return backend.copytree_to_local(src, dst)


def remove(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> None:
    """移除 ``filepath`` 代表的文件.

    Args:
        filepath (str, Path): Path to be removed.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Raises:
        FileNotFoundError: If filepath does not exist, an FileNotFoundError
            will be raised.
        IsADirectoryError: If filepath is a directory, an IsADirectoryError
            will be raised.
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    backend.remove(filepath)


def rmtree(
    dir_path: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> None:
    """递归地删除目录中的文件.

    Args:
        dir_path (str or Path): A directory to be removed.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    backend = get_file_backend(
        dir_path, backend_args=backend_args, enable_singleton=True
    )
    backend.rmtree(dir_path)


def copy_if_symlink_fails(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """创建一个指向 ``src`` 的符号链接, 名为 ``dst``.
    如果创建指向 ``src`` 的符号链接失败, 则将 ``src`` 复制到 ``dst``.

    Args:
        src (str or Path): Create a symbolic link pointing to src.
        dst (str or Path): Create a symbolic link named dst.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return True if successfully create a symbolic link pointing to
        src. Otherwise, return False.
    """
    backend = get_file_backend(src, backend_args=backend_args, enable_singleton=True)
    return backend.copy_if_symlink_fails(src, dst)


def list_dir_or_file(
    dir_path: Union[str, Path],
    list_dir: bool = True,
    list_file: bool = True,
    suffix: Optional[Union[str, Tuple[str]]] = None,
    recursive: bool = False,
    backend_args: Optional[dict] = None,
) -> Iterator[str]:
    """扫描目录, 以任意顺序查找感兴趣的目录或文件.

    Args:
        dir_path (str or Path): Path of the directory.
        list_dir (bool): List the directories. Defaults to True.
        list_file (bool): List the path of files. Defaults to True.
        suffix (str or tuple[str], optional): File suffix that we are
            interested in. Defaults to None.
        recursive (bool): If set to True, recursively scan the directory.
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Yields:
        Iterable[str]: A relative path to ``dir_path``.
    """
    backend = get_file_backend(
        dir_path, backend_args=backend_args, enable_singleton=True
    )
    yield from backend.list_dir_or_file(
        dir_path, list_dir, list_file, suffix, recursive
    )


def generate_presigned_url(
    url: str,
    client_method: str = "get_object",
    expires_in: int = 3600,
    backend_args: Optional[dict] = None,
) -> str:
    """生成视频流的预签名 url, 可以传递给 VideoReader. 目前仅支持 Petrel 后端.

    Args:
        url (str): Url of video stream.
        client_method (str): Method of client, 'get_object' or
            'put_object'. Defaults to 'get_object'.
        expires_in (int): expires, in seconds. Defaults to 3600.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: Generated presigned url.
    """
    backend = get_file_backend(url, backend_args=backend_args, enable_singleton=True)
    return backend.generate_presigned_url(url, client_method, expires_in)


def load(file, file_format=None, backend_args=None, **kwargs):
    """该方法提供了一个统一的 api, 用于从不同后端存储的序列化文件中加载数据.
    如从 json/yaml/pickle 文件中加载数据.
    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and is_str(file):
        file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    handler = file_handlers[file_format]
    if is_str(file):
        file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO(file_backend.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(file_backend.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, backend_args=None, **kwargs):
    """该方法提供了一个统一的 api, 用于将数据以不同的格式保存到不同后端的文件中.
    如将数据保存到 json/yaml/pickle 文件中.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.

    Returns:
        bool: True for success, False otherwise.
    """

    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put(f.getvalue(), file)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
