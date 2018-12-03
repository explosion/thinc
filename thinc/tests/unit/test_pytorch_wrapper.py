# coding: utf8
from __future__ import unicode_literals

from thinc.v2v import Affine
from thinc.neural.optimizers import SGD
import numpy

try:
    import torch.nn
    from thinc.extra.wrappers import PyTorchWrapper
except ImportError:
    PyTorchWrapper = None


def check_learns_zero_output(model, sgd, X, Y):
    """Check we can learn to output a zero vector"""
    Yh, get_dX = model.begin_update(X)
    dYh = (Yh - Y) / Yh.shape[0]
    dX = get_dX(dYh, sgd=sgd)
    prev = numpy.abs(Yh.sum())
    for i in range(100):
        Yh, get_dX = model.begin_update(X)
        total = numpy.abs(Yh.sum())
        dX = get_dX(Yh - Y, sgd=sgd)  # noqa: F841
        assert total < prev
        prev = total


def test_unwrapped(nN=2, nI=3, nO=4):
    if PyTorchWrapper is None:
        return
    model = Affine(nO, nI)
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    sgd = SGD(model.ops, 0.001)
    Y = numpy.zeros((nN, nO), dtype="f")
    check_learns_zero_output(model, sgd, X, Y)


def test_wrapper(nN=2, nI=3, nO=4):
    if PyTorchWrapper is None:
        return
    model = PyTorchWrapper(torch.nn.Linear(nI, nO))
    sgd = SGD(model.ops, 0.001)
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    Y = numpy.zeros((nN, nO), dtype="f")
    Yh, get_dX = model.begin_update(X)
    assert Yh.shape == (nN, nO)
    dYh = (Yh - Y) / Yh.shape[0]
    dX = get_dX(dYh, sgd=sgd)
    assert dX.shape == (nN, nI)
    check_learns_zero_output(model, sgd, X, Y)
