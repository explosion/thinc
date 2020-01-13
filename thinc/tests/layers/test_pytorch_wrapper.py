from thinc.api import Linear, SGD, PyTorchWrapper, xp2torch, torch2xp
import numpy
import pytest


try:
    import torch
    import torch.nn

    has_pytorch = True
except ImportError:
    has_pytorch = False


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


@pytest.mark.skipif(not has_pytorch, reason="needs PyTorch")
@pytest.mark.parametrize("nN,nI,nO", [(2, 3, 4)])
def test_unwrapped(nN, nI, nO):
    model = Linear(nO, nI)
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    sgd = SGD(0.001)
    Y = numpy.zeros((nN, nO), dtype="f")
    check_learns_zero_output(model, sgd, X, Y)


@pytest.mark.skipif(not has_pytorch, reason="needs PyTorch")
@pytest.mark.parametrize("nN,nI,nO", [(2, 3, 4)])
def test_wrapper(nN, nI, nO):
    model = PyTorchWrapper(torch.nn.Linear(nI, nO))
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


@pytest.mark.skipif(not has_pytorch, reason="needs PyTorch")
def test_roundtrip_conversion():
    xp_tensor = numpy.zeros((2, 3), dtype="f")
    torch_tensor = xp2torch(xp_tensor)
    assert isinstance(torch_tensor, torch.Tensor)
    new_xp_tensor = torch2xp(torch_tensor)
    assert numpy.array_equal(xp_tensor, new_xp_tensor)
