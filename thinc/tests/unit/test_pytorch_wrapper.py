from thinc.layers.affine import Affine
from thinc.optimizers import SGD
from thinc.extra.wrappers import PyTorchWrapper
import numpy
import pytest


try:
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
def test_unwrapped(nN=2, nI=3, nO=4):
    model = Affine(nO, nI)
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    sgd = SGD(model.ops, 0.001)
    Y = numpy.zeros((nN, nO), dtype="f")
    check_learns_zero_output(model, sgd, X, Y)


@pytest.mark.skipif(not has_pytorch, reason="needs PyTorch")
def test_wrapper(nN=2, nI=3, nO=4):
    model = PyTorchWrapper(torch.nn.Linear(nI, nO))
    sgd = SGD(model.ops, 0.001)
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    Y = numpy.zeros((nN, nO), dtype="f")
    Yh, get_dX = model.begin_update(X)
    assert Yh.shape == (nN, nO)
    dYh = (Yh - Y) / Yh.shape[0]
    dX = get_dX(dYh)
    model.finish_update(sgd)
    assert dX.shape == (nN, nI)
    check_learns_zero_output(model, sgd, X, Y)
