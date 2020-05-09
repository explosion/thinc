import numpy
import pytest
from thinc.api import SGD, to_categorical, SparseLinear


@pytest.fixture
def instances():
    lengths = numpy.asarray([5, 4], dtype="int32")
    keys = numpy.arange(9, dtype="uint64")
    values = numpy.ones(9, dtype="float32")
    X = (keys, values, lengths)
    y = numpy.asarray([0, 2], dtype="int32")
    return X, to_categorical(y, n_classes=3)


@pytest.fixture
def sgd():
    return SGD(0.001)


def test_basic(instances, sgd):
    X, y = instances
    nr_class = 3
    model = SparseLinear(nr_class).initialize()
    yh, backprop = model.begin_update(X)
    loss1 = ((yh - y) ** 2).sum()
    backprop(yh - y)
    model.finish_update(sgd)
    yh, backprop = model.begin_update(X)
    loss2 = ((yh - y) ** 2).sum()
    assert loss2 < loss1


def test_init():
    model = SparseLinear(3).initialize()
    keys = numpy.ones((5,), dtype="uint64")
    values = numpy.ones((5,), dtype="f")
    lengths = numpy.zeros((2,), dtype="int32")
    lengths[0] = 3
    lengths[1] = 2
    scores, backprop = model.begin_update((keys, values, lengths))
    assert scores.shape == (2, 3)
    d_feats = backprop(scores)
    assert len(d_feats) == 3
