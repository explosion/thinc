from thinc.api import Linear, SGD, PyTorchWrapper, xp2torch, torch2xp, ArgsKwargs
from thinc.util import has_torch
import numpy
import pytest

from ..util import make_tempdir, check_input_converters


def check_learns_zero_output(model, sgd, X, Y):
    """Check we can learn to output a zero vector"""
    Yh, get_dX = model.begin_update(X)
    dYh = (Yh - Y) / Yh.shape[0]
    dX = get_dX(dYh)
    model.finish_update(sgd)
    prev = numpy.abs(Yh.sum())
    for i in range(100):
        Yh, get_dX = model.begin_update(X)
        total = numpy.abs(Yh.sum())
        dX = get_dX(Yh - Y)  # noqa: F841
        model.finish_update(sgd)
        assert total < prev
        prev = total


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
@pytest.mark.parametrize("nN,nI,nO", [(2, 3, 4)])
def test_pytorch_unwrapped(nN, nI, nO):
    model = Linear(nO, nI).initialize()
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    sgd = SGD(0.001)
    Y = numpy.zeros((nN, nO), dtype="f")
    check_learns_zero_output(model, sgd, X, Y)


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
@pytest.mark.parametrize("nN,nI,nO", [(2, 3, 4)])
def test_pytorch_wrapper(nN, nI, nO):
    import torch.nn

    model = PyTorchWrapper(torch.nn.Linear(nI, nO)).initialize()
    sgd = SGD(0.001)
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    Y = numpy.zeros((nN, nO), dtype="f")
    Yh, get_dX = model.begin_update(X)
    assert isinstance(Yh, numpy.ndarray)
    assert Yh.shape == (nN, nO)
    dYh = (Yh - Y) / Yh.shape[0]
    dX = get_dX(dYh)
    model.finish_update(sgd)
    assert dX.shape == (nN, nI)
    check_learns_zero_output(model, sgd, X, Y)
    assert isinstance(model.predict(X), numpy.ndarray)


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
def test_pytorch_roundtrip_conversion():
    import torch

    xp_tensor = numpy.zeros((2, 3), dtype="f")
    torch_tensor = xp2torch(xp_tensor)
    assert isinstance(torch_tensor, torch.Tensor)
    new_xp_tensor = torch2xp(torch_tensor)
    assert numpy.array_equal(xp_tensor, new_xp_tensor)


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
def test_pytorch_wrapper_roundtrip():
    import torch.nn

    model = PyTorchWrapper(torch.nn.Linear(2, 3))
    model_bytes = model.to_bytes()
    PyTorchWrapper(torch.nn.Linear(2, 3)).from_bytes(model_bytes)
    with make_tempdir() as path:
        model_path = path / "model"
        model.to_disk(model_path)
        new_model = PyTorchWrapper(torch.nn.Linear(2, 3)).from_bytes(model_bytes)
        new_model.from_disk(model_path)


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
@pytest.mark.parametrize(
    "data,n_args,kwargs_keys",
    [
        # fmt: off
        (numpy.zeros((2, 3), dtype="f"), 1, []),
        ([numpy.zeros((2, 3), dtype="f"), numpy.zeros((2, 3), dtype="f")], 2, []),
        ((numpy.zeros((2, 3), dtype="f"), numpy.zeros((2, 3), dtype="f")), 2, []),
        ({"a": numpy.zeros((2, 3), dtype="f"), "b": numpy.zeros((2, 3), dtype="f")}, 0, ["a", "b"]),
        (ArgsKwargs((numpy.zeros((2, 3), dtype="f"), numpy.zeros((2, 3), dtype="f")), {"c": numpy.zeros((2, 3), dtype="f")}), 2, ["c"]),
        # fmt: on
    ],
)
def test_pytorch_convert_inputs(data, n_args, kwargs_keys):
    import torch.nn

    model = PyTorchWrapper(torch.nn.Linear(3, 4))
    convert_inputs = model.attrs["convert_inputs"]
    Y, backprop = convert_inputs(model, data, is_train=True)
    check_input_converters(Y, backprop, data, n_args, kwargs_keys, torch.Tensor)
