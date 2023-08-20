import copy
import torch
from TorchTrainer.datasets import (
    BaseDataset,
    Transform,
)
from TorchTrainer.utils.registry import DATASETS, TRANSFORMS
from TorchTrainer.utils.fileio import load


def function_pipeline(data_info):
    return data_info


@TRANSFORMS.register_module()
class CallableTransform:
    def __call__(self, data_info):
        return data_info


@TRANSFORMS.register_module()
class NotCallableTransform:
    pass


@DATASETS.register_module()
class CustomDataset(BaseDataset):
    def __init__(self, data_root="./data/dataset.json", *arg, **args):
        super().__init__(root=data_root, *arg, **args)

    def load_info(self):
        return load("./data/dataset.json")["data_list"]


class TestBaseDataset:
    data_info = dict(img_path="test_img.jpg", height=604, width=640, index=0)
    imgs = torch.rand((2, 3, 32, 32))

    def test_init(self):
        self.pipeline = lambda x: dict(imgs=self.imgs)
        # test the instantiation of self.base_dataset
        dataset = BaseDataset(root="./data/dataset.json")
        assert hasattr(dataset, "metainfo")
        assert hasattr(dataset, "data_list")
        assert hasattr(dataset, "data_address")

        # test the instantiation of self.base_dataset with `serialize=False`
        dataset = BaseDataset(
            root="./data/dataset.json",
            serialize=False,
        )

        assert hasattr(dataset, "metainfo")
        assert hasattr(dataset, "data_list")
        assert hasattr(dataset, "data_address")

        assert len(dataset) == 3
        assert dataset.get_data_info(0) == self.data_info

        # test the instantiation of self.base_dataset without `ann_file`
        dataset = CustomDataset(
            data_root="./data/",
            serialize=False,
        )

    def test_meta(self):
        dataset = BaseDataset(
            root="./data/dataset.json",
        )
        assert dataset.metainfo == dict(
            dataset_type="test_dataset", task_name="test_task", empty_list=[]
        )

        # test dataset.metainfo with passing metainfo into self.base_dataset
        metainfo = dict(classes=("dog",), task_name="new_task")
        dataset1 = BaseDataset(
            root="./data/dataset.json",
        )
        dataset2 = CustomDataset(
            data_root="./data/",
            serialize=False,
        )
        assert dataset1.metainfo == dataset2.metainfo

        # test dataset.metainfo with passing metainfo containing a file into self.base_dataset

    def test_length(self):
        dataset = BaseDataset(
            root="./data/dataset.json",
        )
        assert hasattr(dataset, "data_list")
        assert len(dataset) == 3

    def test_compose(self):
        # test callable transform
        transforms = [function_pipeline]
        compose = Transform(transforms=transforms)
        assert (self.imgs == compose(dict(img=self.imgs))["img"]).all()

        # test transform build from cfg_dict
        transforms = [dict(type="CallableTransform")]
        compose = Transform(transforms=transforms)
        assert (self.imgs == compose(dict(img=self.imgs))["img"]).all()

        # test return None in advance
        transforms = [lambda x: None, function_pipeline]
        compose = Transform(transforms=transforms)
        assert compose(dict(img=self.imgs)) is None

        # when the input transform is None, do nothing
        compose = Transform(None)
        assert (compose(dict(img=self.imgs))["img"] == self.imgs).all()
        compose = Transform([])
        assert (compose(dict(img=self.imgs))["img"] == self.imgs).all()

    def test_getitem(self):
        dataset = BaseDataset(
            root="./data/dataset.json",
        )
        dataset.pipeline = self.pipeline
        assert hasattr(dataset, "data_list")
        assert dataset[0] == dict(imgs=self.imgs)

    def test_get_data_info(self):
        dataset = BaseDataset(
            root="./data/dataset.json",
        )
        assert hasattr(dataset, "data_list")
        assert dataset.get_data_info(0) == self.data_info

    def test_get_subset(self):
        # Test positive int indices.
        dataset = BaseDataset(
            root="./data/dataset.json",
        )
        indices = 2

        dataset_copy = copy.deepcopy(dataset)
        dataset_copy = dataset_copy.get_subset(indices)[0]
        assert len(dataset_copy) == 2
        for i in range(len(dataset_copy)):
            ori_data = dataset[i]
            assert dataset_copy[i] == ori_data

        # Test negative int indices.
        indices = -2
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy = dataset_copy.get_subset(indices)[0]
        assert len(dataset_copy) == 2
        for i in range(len(dataset_copy)):
            ori_data = dataset[len(dataset) + indices + i]
            ori_data["index"] = i
            assert dataset_copy[i] == ori_data

        # If indices is 0, return empty dataset.
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy = dataset_copy.get_subset(0)[0]
        assert len(dataset_copy) == 0

        # Test list indices with positive element.
        indices = [1]
        dataset_copy = copy.deepcopy(dataset)
        ori_data = dataset[1]
        ori_data["index"] = 0
        dataset_copy = dataset_copy.get_subset(indices)[0]
        assert len(dataset_copy) == 1
        assert dataset_copy[0] == ori_data

        # Test list indices with negative element.
        indices = [-1]
        dataset_copy = copy.deepcopy(dataset)
        ori_data = dataset[2]
        ori_data["index"] = 0
        dataset_copy = dataset_copy.get_subset(indices)[0]
        assert len(dataset_copy) == 1
        assert dataset_copy[0] == ori_data

        # Test empty list.
        indices = []
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy = dataset_copy.get_subset(indices)[0]
        assert len(dataset_copy) == 0

        # Test list with multiple positive and negative indices.
        indices = [-1, -2, 0, 1, 2]
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy = dataset_copy.get_subset(indices)[0]
        for i in range(len(dataset_copy)):
            ori_data = dataset[indices[i]]
            ori_data["index"] = i
            assert dataset_copy[i] == ori_data

        # Test list with probability ratio 1.
        indices = [1.0]
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy = dataset_copy.get_subset(indices)[0]
        assert len(dataset_copy) == len(dataset)
        for i in range(len(dataset_copy)):
            ori_data = dataset[i]
            ori_data["index"] = i
            assert dataset_copy[i] == ori_data

        # Test list with multiple  probability ratio.
        indices = [0.4, 0.6]
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy = dataset_copy.get_subset(indices)
        dataset_copy1 = dataset_copy[0]
        dataset_copy2 = dataset_copy[1]

        assert len(dataset_copy1) + len(dataset_copy2) == len(dataset)

        num = 0
        for i in range(len(dataset_copy1)):
            ori_data = dataset[i]
            ori_data["index"] = i
            assert dataset_copy1[i] == ori_data
            num = num + 1
        for i in range(len(dataset_copy2)):
            ori_data = dataset[i + num]
            ori_data["index"] = i
            a = dataset_copy2[i]
            assert dataset_copy2[i] == ori_data


if __name__ == "__main__":
    testBaseDataset = TestBaseDataset()
    testBaseDataset.test_init()
    testBaseDataset.test_meta()
    testBaseDataset.test_length()
    testBaseDataset.test_compose()
    testBaseDataset.test_getitem()
    testBaseDataset.test_get_data_info()
    subsets = testBaseDataset.test_get_subset()
    # TODO: test the filter function
    pass
