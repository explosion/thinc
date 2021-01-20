import pytest
import numpy
from thinc.api import reduce_first, reduce_last
from thinc.types import Ragged

@pytest.fixture
def Xs():
    seqs = [numpy.zeros((10, 8), dtype="f"), numpy.zeros((4, 8), dtype="f")]
    for x in seqs:
        x[0] = 1
        x[-1] = -1
    return seqs


def test_init_reduce_first():
    model = reduce_first()

def test_init_reduce_last():
    model = reduce_last()


def test_reduce_first(Xs):
    model = reduce_first()
    lengths = model.ops.asarray([x.shape[0] for x in Xs], dtype="i")
    X = Ragged(model.ops.flatten(Xs), lengths)
    Y, backprop = model(X, is_train=True)
    assert isinstance(Y, numpy.ndarray)
    assert Y.shape == (len(Xs), Xs[0].shape[1])
    assert Y.dtype == Xs[0].dtype
    assert list(Y[0]) == list(Xs[0][0])
    assert list(Y[1]) == list(Xs[1][0])
    dX = backprop(Y)
    assert dX.dataXd.shape == X.dataXd.shape


def test_reduce_last(Xs):
    model = reduce_last()
    lengths = model.ops.asarray([x.shape[0] for x in Xs], dtype="i")
    X = Ragged(model.ops.flatten(Xs), lengths)
    Y, backprop = model(X, is_train=True)
    assert isinstance(Y, numpy.ndarray)
    assert Y.shape == (len(Xs), Xs[0].shape[1])
    assert Y.dtype == Xs[0].dtype
    assert list(Y[0]) == list(Xs[0][-1])
    assert list(Y[1]) == list(Xs[1][-1])
    dX = backprop(Y)
    assert dX.dataXd.shape == X.dataXd.shape
