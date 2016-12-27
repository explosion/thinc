from ...ops import NumpyOps

import pytest
import numpy


@pytest.fixture
def ops():
    return NumpyOps(reserve=100)


def test_init(ops):
    assert isinstance(ops.data, numpy.ndarray)


def test_reserve():
    # TODO: Not sure how this feature should work still...
    ops = NumpyOps(reserve=100)
    ops.reserve(100)
    data = ops.allocate((10,))
    with pytest.raises(Exception):
        ops.reserve(100)


def test_get_dropout_empty(ops):
    shape = (2,2)
    drop = 0.0
    mask = ops.get_dropout(shape, drop)
    if drop <= 0.0:
        assert mask is None
    else:
        assert mask is not None


def test_get_dropout_not_empty(ops):
    shape = (2,2)
    drop = 0.1
    mask = ops.get_dropout(shape, drop)
    if drop <= 0.0:
        assert mask is None
    else:
        assert mask is not None
    assert mask.shape == shape
    assert all(value >= 0 for value in mask.flatten())
