import warnings
from io import StringIO
from .io import get_text


def list_from_file(
    filename, prefix="", offset=0, max_num=0, encoding="utf-8", backend_args=None
):
    """该函数支持从不同后端存储的文本文件中加载内容并解析为字符串列表.
    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the beginning of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.
        encoding (str): Encoding used to open the file. Defaults to utf-8.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.

    Returns:
        list[str]: A list of strings.
    """

    cnt = 0
    item_list = []

    text = get_text(filename, encoding, backend_args=backend_args)

    with StringIO(text) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip("\n\r"))
            cnt += 1
    return item_list


def dict_from_file(filename, key_type=str, encoding="utf-8", backend_args=None):
    """该函数支持从不同后端存储的文本文件中加载内容并解析为字典.
    每一行应该由空格或制表符分割为两列或更多列. 第一列将被解析为字典键, 而后续列将被解析为字典值.

    Args:
        filename(str): Filename.
        key_type(type): Type of the dict keys. str is user by default and
            type conversion will be performed if specified.
        encoding (str): Encoding used to open the file. Defaults to utf-8.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.

    Returns:
        dict: The parsed contents.
    """

    mapping = {}

    text = get_text(filename, encoding, backend_args=backend_args)

    with StringIO(text) as f:
        for line in f:
            items = line.rstrip("\n").split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
