from ...ops import NumpyOps

import pytest
import numpy


@pytest.fixture
def ops():
    return NumpyOps()


def test_get_dropout_empty(ops):
    shape = (2,2)
    drop = 0.0
    mask = ops.get_dropout_mask(shape, drop)
    if drop <= 0.0:
        assert mask is None
    else:
        assert mask is not None


def test_get_dropout_not_empty(ops):
    shape = (2,2)
    drop = 0.1
    mask = ops.get_dropout_mask(shape, drop)
    if drop <= 0.0:
        assert mask is None
    else:
        assert mask is not None
    assert mask.shape == shape
    assert all(value >= 0 for value in mask.flatten())
