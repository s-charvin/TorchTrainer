import os
import os.path as osp
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union

import TorchTrainer
from .base import BaseStorageBackend


class LocalBackend(BaseStorageBackend):
    """Raw local storage backend."""

    _allow_symlink = True

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, "rb") as f:
            value = f.read()
        return value

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, encoding=encoding) as f:
            text = f.read()
        return text

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        TorchTrainer.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, "wb") as f:
            f.write(obj)

    def put_text(
        self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8"
    ) -> None:
        """Write text to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.
        """
        TorchTrainer.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, "w", encoding=encoding) as f:
            f.write(obj)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return osp.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        return osp.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        return osp.isfile(filepath)

    def join_path(
        self, filepath: Union[str, Path], *filepaths: Union[str, Path]
    ) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        # TODO, if filepath or filepaths are Path, should return Path
        return osp.join(filepath, *filepaths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing.

        Args:
            filepath (str or Path): Path to be read data.
            backend_args (dict, optional): Arguments to instantiate the
                corresponding backend. Defaults to None.
        """
        yield filepath

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a file src to dst and return the destination file.

        src and dst should have the same prefix. If dst specifies a directory,
        the file will be copied into dst using the base filename from src. If
        dst specifies a file that already exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.
        """
        return shutil.copy(src, dst)

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        src and dst should have the same prefix and dst must not already exist.

        TODO: Whether to support dirs_exist_ok parameter.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.
        """
        return shutil.copytree(src, dst)

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a local file src to dst and return the destination file. Same
        as :meth:`copyfile`.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.
        """
        return self.copyfile(src, dst)

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory. Same as
        :meth:`copytree`.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.
        """
        return self.copytree(src, dst)

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy the file src to local dst and return the destination file. Same
        as :meth:`copyfile`.

        If dst specifies a directory, the file will be copied into dst using
        the base filename from src. If dst specifies a file that already
        exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to to local dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.
        """
        return self.copyfile(src, dst)

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a local
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to local dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.
        """
        return self.copytree(src, dst)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f"filepath {filepath} does not exist")

        if self.isdir(filepath):
            raise IsADirectoryError("filepath should be a file")

        os.remove(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.
        """
        shutil.rmtree(dir_path)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        """Create a symbolic link pointing to src named dst.

        If failed to create a symbolic link pointing to src, directly copy src
        to dst instead.

        Args:
            src (str or Path): Create a symbolic link pointing to src.
            dst (str or Path): Create a symbolic link named dst.

        Returns:
            bool: Return True if successfully create a symbolic link pointing
            to src. Otherwise, return False.
        """
        try:
            os.symlink(src, dst)
            return True
        except Exception:
            if self.isfile(src):
                self.copyfile(src, dst)
            else:
                self.copytree(src, dst)
            return False

    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

        Args:
            dir_path (str or Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional): File suffix that we are
                interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the directory.
                Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        if list_dir and suffix is not None:
            raise TypeError("`suffix` should be None when `list_dir` is True")

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith(".") and entry.is_file():
                    rel_path = osp.relpath(entry.path, root)
                    if (suffix is None or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif osp.isdir(entry.path):
                    if list_dir:
                        rel_dir = osp.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(
                            entry.path, list_dir, list_file, suffix, recursive
                        )

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)
