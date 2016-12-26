from ...x2y.vec2vec import Affine

import numpy
import pytest


class MockOps(object):
    def __init__(self):
        pass

    def allocate(self, shape, name=None):
        return numpy.zeros(shape)

    def affine(self, input_bi, W_oi, b_o):
        return numpy.ones((input_bi.shape[0], W_oi.shape[0]))

    def get_dropout(self, shape, drop):
        if drop <= 0:
            return None
        else:
            return numpy.ones(shape) * (1-drop)


@pytest.fixture
def ops():
    return MockOps()


@pytest.fixture
def model(ops):
    return Affine(ops, 10, 6)


def test_init(ops):
    model = Affine(ops, 10, 6)


def test_shape(model):
    assert model.shape == (10, 6)
    assert model.nr_out == 10
    assert model.nr_in == 6


def test_predict_batch(model):
    input_ = model.ops.allocate((5, 6))
    output = model.predict_batch(input_)
    assert output.shape == (5, 10)
    assert all(val == 1. for val in output.flatten())


def test_begin_update(model):
    input_ = model.ops.allocate((5, 6))
    output, finish_update = model.begin_update(input_)
    assert output.shape == (5, 10)
    assert all(val == 1. for val in output.flatten())


def test_predict_batch_not_batch(model):
    input_ = model.ops.allocate((6,))
    with pytest.raises(ValueError):
        model.begin_update(input_)


def test_predict_update_dim_mismatch(model):
    input_ = model.ops.allocate((10, 5))
    with pytest.raises(ValueError):
        model.begin_update(input_)


def test_begin_update_not_batch(model):
    input_ = model.ops.allocate((6,))
    with pytest.raises(ValueError):
        model.begin_update(input_)


def test_begin_update_dim_mismatch(model):
    input_ = model.ops.allocate((10, 5))
    with pytest.raises(ValueError):
        model.begin_update(input_)
