import copy
import os.path as osp
from torchTraining import Config, ConfigDict
from torchTraining.utils.fileio import dump, load


class TestConfig:
    data_path = osp.join(osp.dirname(__file__), "data")

    def test_init(self):
        cfg = Config()
        assert cfg.filename is None
        assert cfg.text == ""
        assert len(cfg) == 0
        assert cfg._cfg_dict == {}

        # 测试 cfg_dict 参数, 必须输入字典否则出错
        print(Config({"a": [0, 1]}).pretty_text)

        # 测试 filename 参数, 此文件应存储了参数配置的文本形式信息
        cfg_dict = dict(item1=[1, 2], item2=dict(a=0), item3=True, item4="test")
        cfg_file = f"{self.data_path}/base.py"
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file).read()

    def test_fromfile(self):
        # 测试加载配置文件的功能
        cfg_file = f"{self.data_path}/base.py"
        print(Config.fromfile(cfg_file).pretty_text + "\n\n\n")
        cfg_file = f"{self.data_path}/base.json"
        print(Config.fromfile(cfg_file).pretty_text + "\n\n\n")
        cfg_file = f"{self.data_path}/base.yaml"
        print(Config.fromfile(cfg_file).pretty_text + "\n\n\n")

    def test_fromstring(self):
        cfg_file = f"{self.data_path}/base.py"
        file_format = osp.splitext(cfg_file)[-1]
        in_cfg = Config.fromfile(cfg_file)
        cfg_str = open(cfg_file).read()
        out_cfg = Config.fromstring(cfg_str, file_format)
        assert in_cfg._cfg_dict == out_cfg._cfg_dict
        assert in_cfg.pretty_text == out_cfg.pretty_text

        # # 错误示例(字符串加载时不能使用基本变量):

        # cfg_file = f"{self.data_path}/base.json"
        # file_format = osp.splitext(cfg_file)[-1]
        # in_cfg = Config.fromfile(cfg_file)
        # cfg_str = open(cfg_file).read()
        # out_cfg = Config.fromstring(cfg_str, file_format)
        # assert in_cfg._cfg_dict == out_cfg._cfg_dict
        # assert in_cfg.pretty_text == out_cfg.pretty_text

    def test_magic_methods(self):
        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)
        # len(cfg)
        print(f"length: {len(cfg)}")
        # cfg.keys()
        print(f"keys: {cfg.keys()}")
        print(f"_cfg_dict.keys(): {cfg._cfg_dict.keys()}")
        # cfg.values()
        print(f"values: \n")
        for value in cfg.values():
            print(f"\t value: {value}")

        # cfg.update()
        cfg.update(dict(update_item1=0))
        assert cfg.update_item1 == 0
        cfg.update(dict(update_item1=dict(a=1)))
        assert cfg.update_item1.a == 1
        # test __setattr__
        cfg = Config()
        cfg.setitem1 = [1, 2]
        cfg.setitem2 = {"a": 0}
        cfg["setitem3"] = {"a": {"b": None}}
        assert cfg._cfg_dict["setitem1"] == [1, 2]
        assert cfg.setitem1 == [1, 2]
        assert cfg._cfg_dict["setitem2"] == {"a": 0}
        assert cfg.setitem2.a == 0
        assert cfg._cfg_dict["setitem3"] == {"a": {"b": None}}
        assert cfg.setitem3.a.b is None

    def test_merge_from_dict(self):
        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)

        input_options = {"item8": "", "item9": {"a": 5}, "item10": {"a": 10}}
        cfg.merge_from_dict(input_options)
        assert cfg.item8 == ""
        assert cfg.item9 == {"a": 5}
        assert cfg.item10 == {"a": 10}

        # Allow list keys
        input_options = {"list_item.0.a": 1, "list_item.1.b": 1}
        cfg.merge_from_dict(input_options, allow_list_keys=True)
        # TODO: 
        # assert cfg.list_item == [{"a": 1}, {"b": 1, "c": 0}]

        # allow_list_keys is False
        input_options = {"list_item.0.a": 1, "list_item.1.b": 1}
        cfg.merge_from_dict(input_options, allow_list_keys=False)
        assert cfg.list_item == {'0': {'a': 1}, '1': {'b': 1}}
        # Overflowed index number
        input_options = {"list_item.2.a": 1}
        cfg.merge_from_dict(input_options, allow_list_keys=True)

    def test_dict_to_config_dict(self):
        cfg_dict = dict(a=1, b=dict(c=dict()), d=[dict(e=dict(f=(dict(g=1), [])))])
        cfg_dict = Config._dict_to_config_dict(cfg_dict)
        assert isinstance(cfg_dict, ConfigDict)
        assert isinstance(cfg_dict.a, int)
        assert isinstance(cfg_dict.b, ConfigDict)
        assert isinstance(cfg_dict.b.c, ConfigDict)
        assert isinstance(cfg_dict.d, list)
        assert isinstance(cfg_dict.d[0], ConfigDict)
        assert isinstance(cfg_dict.d[0].e, ConfigDict)
        assert isinstance(cfg_dict.d[0].e.f, tuple)
        assert isinstance(cfg_dict.d[0].e.f[0], ConfigDict)
        assert isinstance(cfg_dict.d[0].e.f[1], list)

    def test_dump(self):
        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)
        dump_py = "./dump_config.py"

        cfg.dump(dump_py)
        assert cfg.dump() == cfg.pretty_text
        assert open(dump_py).read() == cfg.pretty_text

        # test dump json/yaml.
        cfg_file = f"{self.data_path}/base.json"
        cfg = Config.fromfile(cfg_file)
        dump_json = "./dump_config.json"
        cfg.dump(dump_json)

        with open(dump_json) as f:
            assert f.read() == cfg.dump()

        # test dump and dump to pickle

        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)
        text_cfg_filename = "./text_dump_config.py"
        cfg.dump(text_cfg_filename)

        text_cfg = Config.fromfile(text_cfg_filename)
        assert text_cfg._cfg_dict == cfg._cfg_dict

        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)
        pkl_cfg_filename = "./dump_config_pickle.pkl"
        dump(cfg, pkl_cfg_filename)
        pkl_cfg = load(pkl_cfg_filename)
        assert pkl_cfg._cfg_dict == cfg._cfg_dict

        # Test dump config from dict.
        cfg_dict = dict(a=1, b=2)
        cfg = Config(cfg_dict)
        assert cfg.pretty_text == cfg.dump()
        # Test dump python format config.
        dump_file = "./dump_from_dict.py"
        cfg.dump(dump_file)
        with open(dump_file) as f:
            assert f.read() == "a = 1\nb = 2\n"
        # Test dump json format config.
        dump_file = "./dump_from_dict.json"
        cfg.dump(dump_file)
        with open(dump_file) as f:
            assert f.read() == '{"a": 1, "b": 2}'
        # Test dump yaml format config.
        dump_file = "./dump_from_dict.yaml"
        cfg.dump(dump_file)
        with open(dump_file) as f:
            assert f.read() == "a: 1\nb: 2\n"

    def test_pretty_text(self):
        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)
        text_cfg_filename = "./_text_config.py"
        with open(text_cfg_filename, "w") as f:
            f.write(cfg.pretty_text)
        text_cfg = Config.fromfile(text_cfg_filename)
        assert text_cfg._cfg_dict == cfg._cfg_dict

    def test_repr(self):
        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)
        tmp_txt = "./tmp.txt"
        with open(tmp_txt, "w") as f:
            print(cfg, file=f)
        with open(tmp_txt) as f:
            assert (
                f.read().strip() == f"Config (path: {cfg.filename}): "
                f"{cfg._cfg_dict.__repr__()}"
            )

    def test_validate_py_syntax(self):
        tmp_cfg = "./tmp_config.py"
        # with open(tmp_cfg, "w") as f:
        #     f.write("dict(a=1,b=2.c=3)")
        # # Incorrect point in dict will cause error

        # Config._validate_py_syntax(tmp_cfg)
        # with open(tmp_cfg, "w") as f:
        #     f.write("[dict(a=1, b=2, c=(1, 2)]")
        # # Imbalance bracket will cause error

        # Config._validate_py_syntax(tmp_cfg)
        # with open(tmp_cfg, "w") as f:
        #     f.write("dict(a=1,b=2\nc=3)")
        # # Incorrect feed line in dict will cause error

        # Config._validate_py_syntax(tmp_cfg)

    def test_deepcopy(self):
        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)
        new_cfg = copy.deepcopy(cfg)

        assert isinstance(new_cfg, Config)
        assert new_cfg._cfg_dict == cfg._cfg_dict
        assert new_cfg._cfg_dict is not cfg._cfg_dict
        assert new_cfg.filename == cfg.filename
        assert new_cfg.text == cfg.text

    def test_copy(self):
        cfg_file = f"{self.data_path}/config.py"
        cfg = Config.fromfile(cfg_file)
        new_cfg = copy.deepcopy(cfg)

        assert isinstance(new_cfg, Config)
        assert new_cfg._cfg_dict == cfg._cfg_dict
        assert new_cfg._cfg_dict == cfg._cfg_dict
        assert new_cfg.filename == cfg.filename
        assert new_cfg.text == cfg.text


if __name__ == "__main__":
    test_config = TestConfig()
    test_config.test_init()
    test_config.test_fromfile()
    test_config.test_fromstring()
    test_config.test_magic_methods()
    test_config.test_merge_from_dict()
    test_config.test_dict_to_config_dict()
    test_config.test_dump()
    test_config.test_pretty_text()
    test_config.test_repr()
    test_config.test_validate_py_syntax()
    test_config.test_deepcopy()
    test_config.test_copy()
