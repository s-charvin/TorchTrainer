
import itertools
import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.nn.init import constant_
import tempfile
from TorchTrainer.model import (
    BaseDataPreprocessor,
    BaseModel,
    ImgDataPreprocessor,
    BaseModule,
)
from TorchTrainer.optim import OptimWrapper
from TorchTrainer.structures import InstanceData
from TorchTrainer.utils.registry import MODELS, Registry, build_from_cfg
from torch.testing import assert_allclose

dtypes_to_test = [torch.float16, torch.float32, torch.float64, torch.half]

devices_to_test = ["cpu", torch.device("cpu")]
if torch.cuda.is_available():
    devices_to_test += ["cuda", 0, torch.device("cuda")]


class TestBaseDataPreprocessor:
    def test_init(self):
        # test data_preprocessor without block
        data_preprocessor = BaseDataPreprocessor()
        assert data_preprocessor._device == torch.device("cpu")
        assert data_preprocessor._non_blocking == False
        # test data_preprocessor with block
        data_preprocessor = BaseDataPreprocessor(True)
        assert data_preprocessor._device == torch.device("cpu")
        assert data_preprocessor._non_blocking == True

    def test_forward(self):
        data_preprocessor = BaseDataPreprocessor()
        input1 = torch.randn(1, 3, 5)
        input2 = torch.randn(1, 3, 5)
        label1 = torch.randn(1)
        label2 = torch.randn(1)

        # Test with list of data inputs and data samples
        data = dict(inputs=[input1, input2], data_sample=[label1, label2])
        output = data_preprocessor(data)
        batch_inputs, batch_labels = output["inputs"], output["data_sample"]

        assert torch.is_floating_point(batch_inputs[0])
        assert batch_inputs[0].shape == (1, 3, 5)

        assert_allclose(input1, batch_inputs[0])
        assert_allclose(input2, batch_inputs[1])
        assert_allclose(label1, batch_labels[0])
        assert_allclose(label2, batch_labels[1])

        # Test with tuple of batch inputs and batch data samples
        data = (torch.stack([input1, input2]), (label1, label2))
        batch_inputs, batch_labels = data_preprocessor(data)

        assert torch.is_floating_point(batch_inputs)
        assert batch_inputs[0].shape == (1, 3, 5)
        assert batch_inputs[1].shape == (1, 3, 5)
        assert torch.is_floating_point(batch_inputs[0])

        # Test cuda forward
        if torch.cuda.is_available():
            # Test with list of data inputs and data samples
            data = dict(inputs=[input1, input2], data_sample=[label1, label2])
            data_preprocessor = data_preprocessor.cuda()
            output = data_preprocessor(data)
            batch_inputs, batch_labels = output["inputs"], output["data_sample"]
            assert torch.is_floating_point(batch_inputs[0])
            assert batch_inputs[0].device.type == "cuda"

            # Fallback to test with cpu.
            data_preprocessor = data_preprocessor.cpu()
            output = data_preprocessor(data)
            batch_inputs, batch_labels = output["inputs"], output["data_sample"]
            assert torch.is_floating_point(batch_inputs[0])
            assert batch_inputs[0].device.type == "cpu"

            # Test `data_preprocessor` can be moved to cuda again.
            data_preprocessor = data_preprocessor.to("cuda:0")
            output = data_preprocessor(data)
            batch_inputs, batch_labels = output["inputs"], output["data_sample"]
            assert torch.is_floating_point(batch_inputs[0])
            assert batch_inputs[0].device.type == "cuda"

            # device of `data_preprocessor` is cuda, output should be
            # cuda tensor.
            assert batch_inputs[0].device.type == "cuda"
            assert batch_labels[0].device.type == "cuda"

        # Test data_preprocessor with string value
        data = dict(string="abc")
        data_preprocessor(data)


class TestImgDataPreprocessor(TestBaseDataPreprocessor):
    def test_init(self):
        # Initiate processor without arguments
        data_processor = ImgDataPreprocessor()
        assert not data_processor._channel_conversion
        assert not hasattr(data_processor, "mean")
        assert not hasattr(data_processor, "std")
        assert data_processor.pad_size_divisor == 1
        assert_allclose(data_processor.pad_value, torch.tensor(0))

        # Initiate model with bgr2rgb, mean, std .etc..
        data_processor = ImgDataPreprocessor(
            bgr_to_rgb=True,
            mean=[0, 0, 0],
            std=[255, 255, 255],
            pad_size_divisor=16,
            pad_value=10,
        )
        assert data_processor._enable_normalize
        assert data_processor._channel_conversion
        assert_allclose(data_processor.mean, torch.tensor([0, 0, 0]).view(-1, 1, 1))
        assert_allclose(
            data_processor.std, torch.tensor([255, 255, 255]).view(-1, 1, 1)
        )
        assert_allclose(data_processor.pad_value, torch.tensor(10))
        assert data_processor.pad_size_divisor == 16
        try:
            ImgDataPreprocessor(mean=(1, 2), std=(1, 2, 3))
        except AssertionError:
            pass

        try:
            ImgDataPreprocessor(mean=(1, 2, 3), std=(1, 2))
        except AssertionError:
            pass
        try:
            ImgDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)
        except AssertionError:
            pass

        try:
            ImgDataPreprocessor(
                bgr_to_rgb=True,
                mean=None,
                std=[255, 255, 255],
                pad_size_divisor=16,
                pad_value=10,
            )
        except AssertionError:
            pass

        data_processor = ImgDataPreprocessor(
            bgr_to_rgb=True, pad_size_divisor=16, pad_value=10
        )
        assert not data_processor._enable_normalize

    def test_forward(self):
        # Test `pad_value`, `to_rgb`, `pad_size_divisor`.
        data_preprocessor = ImgDataPreprocessor(
            mean=[127.5],
            std=[1, 2, 3],
            pad_size_divisor=16,
            pad_value=10,
            rgb_to_bgr=True,
        )
        inputs1 = torch.randn(3, 10, 10)
        inputs2 = torch.randn(3, 15, 15)
        data_sample1 = InstanceData(bboxes=torch.randn(5, 4))
        data_sample2 = InstanceData(bboxes=torch.randn(5, 4))

        data = dict(
            inputs=[inputs1.clone(), inputs2.clone()],
            data_sample=[data_sample1.clone(), data_sample2.clone()],
        )

        std = torch.tensor([1, 2, 3]).view(-1, 1, 1)
        target_inputs1 = (inputs1.clone()[[2, 1, 0], ...] - 127.5) / std
        target_inputs2 = (inputs2.clone()[[2, 1, 0], ...] - 127.5) / std

        target_inputs1 = F.pad(target_inputs1, (0, 6, 0, 6), value=10)
        target_inputs2 = F.pad(target_inputs2, (0, 1, 0, 1), value=10)

        target_inputs = [target_inputs1, target_inputs2]
        output = data_preprocessor(data, True)
        inputs, data_samples = output["inputs"], output["data_sample"]
        assert torch.is_floating_point(inputs)

        target_data_samples = [data_sample1, data_sample2]
        for input_, data_sample, target_input, target_data_sample in zip(
            inputs, data_samples, target_inputs, target_data_samples
        ):
            assert_allclose(input_, target_input)
            assert_allclose(data_sample.bboxes, target_data_sample.bboxes)

        # Test image without normalization.
        data_preprocessor = ImgDataPreprocessor(
            pad_size_divisor=16,
            pad_value=10,
            rgb_to_bgr=True,
        )
        target_inputs1 = inputs1.clone()[[2, 1, 0], ...]
        target_inputs2 = inputs2.clone()[[2, 1, 0], ...]
        target_inputs1 = F.pad(target_inputs1, (0, 6, 0, 6), value=10)
        target_inputs2 = F.pad(target_inputs2, (0, 1, 0, 1), value=10)

        target_inputs = [target_inputs1, target_inputs2]
        output = data_preprocessor(data, True)
        inputs, data_samples = output["inputs"], output["data_sample"]
        assert torch.is_floating_point(inputs)

        target_data_samples = [data_sample1, data_sample2]
        for input_, data_sample, target_input, target_data_sample in zip(
            inputs, data_samples, target_inputs, target_data_samples
        ):
            assert_allclose(input_, target_input)
            assert_allclose(data_sample.bboxes, target_data_sample.bboxes)

        # Test gray image with 3 dim mean will raise error
        data_preprocessor = ImgDataPreprocessor(
            mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)
        )
        data = dict(inputs=[torch.ones(10, 10), torch.ones(10, 10)], data_sample=None)
        try:
            data_preprocessor(data)
        except AssertionError:
            pass

        try:
            data = dict(
                inputs=[torch.ones(10, 10), torch.ones(10, 10)], data_sample=None
            )
            data_preprocessor(data)
        except AssertionError:
            pass

        # Test stacked batch inputs and batch data samples
        data_preprocessor = ImgDataPreprocessor(
            mean=(127.5, 127.5, 127.5),
            std=(127.5, 127.5, 127.5),
            rgb_to_bgr=True,
            pad_size_divisor=16,
        )
        _batch_inputs = torch.randn(2, 3, 10, 10)
        _batch_labels = [torch.randn(1), torch.randn(1)]
        data = dict(inputs=_batch_inputs, data_sample=_batch_labels)
        output = data_preprocessor(data)
        inputs, data_samples = output["inputs"], output["data_sample"]
        target_batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
        target_batch_inputs = (target_batch_inputs - 127.5) / 127.5
        target_batch_inputs = F.pad(target_batch_inputs, (0, 6, 0, 6), value=0)
        assert inputs.shape == torch.Size([2, 3, 16, 16])
        assert torch.is_floating_point(inputs)
        assert_allclose(target_batch_inputs, inputs)

        # Test batch inputs without convert channel order and pad
        data_preprocessor = ImgDataPreprocessor(
            mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)
        )
        _batch_inputs = torch.randn(2, 3, 10, 10)
        _batch_labels = [torch.randn(1), torch.randn(1)]
        data = dict(inputs=_batch_inputs, data_sample=_batch_labels)
        output = data_preprocessor(data)
        inputs, data_samples = output["inputs"], output["data_sample"]
        target_batch_inputs = (_batch_inputs - 127.5) / 127.5
        assert inputs.shape == torch.Size([2, 3, 10, 10])
        assert torch.is_floating_point(inputs)
        assert_allclose(target_batch_inputs, inputs)

        # Test empty `data_sample`
        data = dict(inputs=[inputs1.clone(), inputs2.clone()], data_sample=None)
        output = data_preprocessor(data, True)
        inputs, data_samples = output["inputs"], output["data_sample"]
        assert data_samples is None
        assert torch.is_floating_point(inputs)


COMPONENTS = Registry("component")
FOOMODELS = Registry("model")


@COMPONENTS.register_module()
class FooConv1d(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1d = nn.Conv1d(4, 1, 4)

    def forward(self, x):
        return self.conv1d(x)


@COMPONENTS.register_module()
class FooConv2d(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv2d = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        return self.conv2d(x)


@COMPONENTS.register_module()
class FooLinear(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)


@COMPONENTS.register_module()
class FooLinearConv1d(BaseModule):
    def __init__(self, linear=None, conv1d=None, init_cfg=None):
        super().__init__(init_cfg)
        if linear is not None:
            self.linear = build_from_cfg(linear, COMPONENTS)
        if conv1d is not None:
            self.conv1d = build_from_cfg(conv1d, COMPONENTS)

    def forward(self, x):
        x = self.linear(x)
        return self.conv1d(x)


@FOOMODELS.register_module()
class FooModel(BaseModule):
    def __init__(
        self,
        component1=None,
        component2=None,
        component3=None,
        component4=None,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg)
        if component1 is not None:
            self.component1 = build_from_cfg(component1, COMPONENTS)
        if component2 is not None:
            self.component2 = build_from_cfg(component2, COMPONENTS)
        if component3 is not None:
            self.component3 = build_from_cfg(component3, COMPONENTS)
        if component4 is not None:
            self.component4 = build_from_cfg(component4, COMPONENTS)

        self.reg = nn.Linear(3, 4)


class TestModule:
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_module = BaseModule()
        self.model_cfg = dict(
            type="FooModel",
            init_cfg=[
                dict(type="Constant", val=1, bias=2, layer="Linear"),
                dict(type="Constant", val=3, bias=4, layer="Conv1d"),
                dict(type="Constant", val=5, bias=6, layer="Conv2d"),
            ],
            component1=dict(type="FooConv1d"),
            component2=dict(type="FooConv2d"),
            component3=dict(type="FooLinear"),
            component4=dict(
                type="FooLinearConv1d",
                linear=dict(type="FooLinear"),
                conv1d=dict(type="FooConv1d"),
            ),
        )

        self.model = build_from_cfg(self.model_cfg, FOOMODELS)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_is_init(self):
        assert self.base_module.is_init is False

    def test_init_weights(self):
        # Test init_weights with layer attribute.
        self.model.init_weights()

        assert torch.equal(
            self.model.component1.conv1d.weight,
            torch.full(self.model.component1.conv1d.weight.shape, 3.0),
        )
        assert torch.equal(
            self.model.component1.conv1d.bias,
            torch.full(self.model.component1.conv1d.bias.shape, 4.0),
        )
        assert torch.equal(
            self.model.component2.conv2d.weight,
            torch.full(self.model.component2.conv2d.weight.shape, 5.0),
        )
        assert torch.equal(
            self.model.component2.conv2d.bias,
            torch.full(self.model.component2.conv2d.bias.shape, 6.0),
        )
        assert torch.equal(
            self.model.component3.linear.weight,
            torch.full(self.model.component3.linear.weight.shape, 1.0),
        )
        assert torch.equal(
            self.model.component3.linear.bias,
            torch.full(self.model.component3.linear.bias.shape, 2.0),
        )
        assert torch.equal(
            self.model.component4.linear.linear.weight,
            torch.full(self.model.component4.linear.linear.weight.shape, 1.0),
        )
        assert torch.equal(
            self.model.component4.linear.linear.bias,
            torch.full(self.model.component4.linear.linear.bias.shape, 2.0),
        )
        assert torch.equal(
            self.model.component4.conv1d.conv1d.weight,
            torch.full(self.model.component4.conv1d.conv1d.weight.shape, 3.0),
        )
        assert torch.equal(
            self.model.component4.conv1d.conv1d.bias,
            torch.full(self.model.component4.conv1d.conv1d.bias.shape, 4.0),
        )
        assert torch.equal(
            self.model.reg.weight, torch.full(self.model.reg.weight.shape, 1.0)
        )
        assert torch.equal(
            self.model.reg.bias, torch.full(self.model.reg.bias.shape, 2.0)
        )
        # Test init_weights with layer attribute and name attribute with duplicate information.
        model_cfg = dict(
            type="FooModel",
            init_cfg=[
                dict(type="Constant", val=1, bias=2, layer="Linear"),
                dict(type="Constant", val=3, bias=4, layer="Conv1d"),
                dict(type="Constant", val=5, bias=6, layer="Conv2d"),
                dict(type="Constant", val=7, bias=8, name="component1"),
                dict(type="Constant", val=9, bias=10, name="component2.conv2d"),
            ],
            component1=dict(type="FooConv1d"),
            component2=dict(type="FooConv2d"),
            component3=dict(type="FooLinear"),
            component4=dict(
                type="FooLinearConv1d",
                linear=dict(type="FooLinear"),
                conv1d=dict(type="FooConv1d"),
            ),
        )

        self.model = build_from_cfg(model_cfg, FOOMODELS)
        self.model.init_weights()
        assert torch.equal(
            self.model.component1.conv1d.weight,
            torch.full(self.model.component1.conv1d.weight.shape, 7.0),
        )
        assert torch.equal(
            self.model.component2.conv2d.weight,
            torch.full(self.model.component2.conv2d.weight.shape, 9.0),
        )

        # Test init_weights  from Pretrained model weights

        class CustomLinear(BaseModule):
            def __init__(self, init_cfg=None):
                super().__init__(init_cfg)
                self.linear = nn.Linear(1, 1)

            def init_weights(self):
                constant_(self.linear.weight, 1)
                constant_(self.linear.bias, 2)

        @FOOMODELS.register_module()
        class PratrainedModel(FooModel):
            def __init__(
                self,
                component1=None,
                component2=None,
                component3=None,
                component4=None,
                init_cfg=None,
            ) -> None:
                super().__init__(
                    component1, component2, component3, component4, init_cfg
                )
                self.linear = CustomLinear()

        checkpoint_path = osp.join(self.temp_dir.name, "test.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg["type"] = "PratrainedModel"
        model_cfg["init_cfg"] = dict(type="Pretrained", checkpoint=checkpoint_path)
        model = FOOMODELS.build(model_cfg)
        ori_layer_weight = model.linear.linear.weight.clone()
        ori_layer_bias = model.linear.linear.bias.clone()
        model.init_weights()

        assert (ori_layer_weight != model.linear.linear.weight).any()
        assert (ori_layer_bias != model.linear.linear.bias).any()

        class FakeDDP(nn.Module):
            def __init__(self, module) -> None:
                super().__init__()
                self.module = module

        # Test initialization of nested modules in (分布式数据并行)DDPModule which define `init_weights`.

        model = FOOMODELS.build(self.model_cfg)
        model.ddp = FakeDDP(CustomLinear())
        model.init_weights()
        assert (model.ddp.module.linear.weight == 1).all()
        assert (model.ddp.module.linear.bias == 2).all()


def list_product(*args):
    return list(itertools.product(*args))


@MODELS.register_module()
class CustomDataPreprocessor(BaseDataPreprocessor):
    def forward(self, data, training=False):
        if training:
            return 1
        else:
            return 2


@MODELS.register_module()
class ToyModel(BaseModel):
    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        self.backbone = nn.Sequential()
        self.backbone.add_module("conv1", nn.Conv2d(3, 6, 5, padding=2))
        self.backbone.add_module("pool", nn.MaxPool2d(2, 2))
        self.backbone.add_module("conv2", nn.Conv2d(6, 16, 5, padding=2))
        self.backbone.add_module("flatten", nn.Flatten())
        self.backbone.add_module("fc1", nn.Linear(16 * 5 * 5, 120))
        self.backbone.add_module("fc2", nn.Linear(120, 84))
        self.backbone.add_module("fc3", nn.Linear(84, 10))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, data_sample=None, mode="tensor"):
        if mode == "tensor":
            return self.backbone(inputs)
        elif mode == "predict":
            feats = self.backbone(inputs)
            predictions = torch.argmax(feats, 1)
            return predictions
        elif mode == "loss":
            feats = self.backbone(inputs)
            loss = self.criterion(feats, data_sample)
            return dict(loss=loss)


@MODELS.register_module()
class NestedModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.toy_model = ToyModel()

    def forward(self):
        pass


class TestModel:
    def test_init(self):
        # test model without `data_preprocessor`.
        model = ToyModel()
        assert isinstance(model.data_preprocessor, BaseDataPreprocessor)

        # test model with `data_preprocessor` config.
        data_preprocessor = dict(type="CustomDataPreprocessor")
        model = ToyModel(data_preprocessor=data_preprocessor)
        assert isinstance(model.data_preprocessor, CustomDataPreprocessor)
        assert model.data_preprocessor(1, training=True) == 1
        assert model.data_preprocessor(1, training=False) == 2

        # test model with `data_preprocessor` Class.
        data_preprocessor = CustomDataPreprocessor()
        model = ToyModel(data_preprocessor=data_preprocessor)
        assert model.data_preprocessor == data_preprocessor

        # initiate model with error type `data_preprocessor`.
        try:
            model = ToyModel(data_preprocessor=[data_preprocessor])
        except TypeError:
            pass

    def test_parse_losses(self):
        model = ToyModel()

        # parse custom loss
        losses = dict(
            loss_cls=torch.tensor(1, dtype=torch.float32),
            loss_list=[
                torch.tensor(2, dtype=torch.float32),
                torch.tensor(3, dtype=torch.float32),
            ],
        )
        parsed_losses, log_vars = model.parse_losses(losses)

        # target parsed losses
        target_parsed_losses = torch.tensor(6, dtype=torch.float32)
        targe_log_vars = dict(
            loss=torch.tensor(6, dtype=torch.float32),
            loss_cls=torch.tensor(1, dtype=torch.float32),
            loss_list=torch.tensor(5, dtype=torch.float32),
        )

        # test parsed losses
        assert_allclose(parsed_losses, target_parsed_losses)
        for key in log_vars:
            assert key in targe_log_vars
            assert_allclose(log_vars[key], targe_log_vars[key])
        # test model with error type `losses`.
        losses["error_key"] = dict()
        try:
            model.parse_losses(losses)
        except TypeError:
            pass

    def test_train_step(self):
        model = ToyModel()
        ori_conv_weight = model.backbone.conv1.weight.clone()

        optimizer = SGD(model.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        data_sample = torch.tensor([1, 2])
        data = dict(inputs=torch.randn((2, 3, 10, 10)), data_sample=data_sample)
        log_vars = model.train_step(data, optim_wrapper)

        assert not (torch.equal(ori_conv_weight, model.backbone.conv1.weight))
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_step(self):
        model = ToyModel()
        data_sample = torch.tensor([1, 2])
        data = dict(inputs=torch.randn((2, 3, 10, 10)), data_sample=data_sample)

        out = model.val_step(data)
        assert isinstance(out, torch.Tensor)

    def test_test_step(self):
        model = ToyModel()
        data_sample = torch.tensor([1, 2])
        data = dict(inputs=torch.randn((2, 3, 10, 10)), data_sample=data_sample)

        out = model.test_step(data)
        assert isinstance(out, torch.Tensor)

    def test_cuda(self):
        model = ToyModel().cuda()
        data_sample = torch.tensor([1, 2])
        data = dict(inputs=torch.randn((2, 3, 10, 10)).cuda(), data_sample=data_sample)

        out = model.val_step(data)
        assert out.device.type == "cuda"

        model = NestedModel()
        assert model.data_preprocessor._device == torch.device("cpu")
        assert model.toy_model.data_preprocessor._device == torch.device("cpu")

        model.cuda()
        assert model.data_preprocessor._device == torch.device("cuda")
        assert model.toy_model.data_preprocessor._device == torch.device("cuda")

    def test_to(self):
        model = ToyModel().to(torch.cuda.current_device())
        data_sample = torch.tensor([1, 2])
        data = dict(
            inputs=torch.randn((2, 3, 10, 10)).to("cuda:0"), data_sample=data_sample
        )

        out = model.val_step(data)
        assert out.device.type == "cuda"

        model = NestedModel()
        assert model.data_preprocessor._device == torch.device("cpu")
        assert model.toy_model.data_preprocessor._device == torch.device("cpu")

        model.to("cuda")
        assert model.data_preprocessor._device == torch.device("cuda")
        assert model.toy_model.data_preprocessor._device == torch.device("cuda")

        model.to()
        assert model.data_preprocessor._device == torch.device("cuda")
        assert model.toy_model.data_preprocessor._device == torch.device("cuda")

    def test_to_device(self):
        devices = ["cuda", 0, torch.device("cuda"), "cpu", torch.device("cpu")]
        for device in devices:
            model = ToyModel().to(device)
            assert all(
                p.device.type == torch.device(device).type for p in model.parameters()
            ) and model.data_preprocessor._device == torch.device(device)

    def test_to_dtype(self):
        dtypes = [torch.float16, torch.float32, torch.float64, torch.half]
        for dtype in dtypes:
            model = ToyModel().to(dtype)
            assert all(p.dtype == dtype for p in model.parameters())

    def test_to_device_and_dtype(self):
        devices = ["cuda", 0, torch.device("cuda"), "cpu", torch.device("cpu")]
        dtypes = [torch.float16, torch.float32, torch.float64, torch.half]
        for device, dtype in list_product(devices, dtypes):
            model = ToyModel().to(device=device, dtype=dtype)
            assert (
                all(p.dtype == dtype for p in model.parameters())
                and model.data_preprocessor._device == torch.device(device)
                and all(
                    p.device.type == torch.device(device).type
                    for p in model.parameters()
                )
            )


if __name__ == "__main__":
    base_data_preprocessor_test = TestBaseDataPreprocessor()
    base_data_preprocessor_test.test_init()
    base_data_preprocessor_test.test_forward()

    img_data_preprocessor_test = TestImgDataPreprocessor()
    img_data_preprocessor_test.test_init()
    img_data_preprocessor_test.test_forward()

    module_test = TestModule()
    module_test.setUp()
    module_test.test_is_init()
    module_test.test_init_weights()
    module_test.tearDown()

    model_test = TestModel()
    model_test.test_init()
    model_test.test_parse_losses()
    model_test.test_train_step()
    model_test.test_val_step()
    model_test.test_test_step()
    model_test.test_cuda()
    model_test.test_to()
    model_test.test_to_device()
    model_test.test_to_dtype()
    model_test.test_to_device_and_dtype()
