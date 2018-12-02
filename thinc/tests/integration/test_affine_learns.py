# coding: utf8
from __future__ import division, print_function

import pytest
import numpy as np
import random
from numpy.testing import assert_allclose

from ...neural.optimizers import SGD
from ...neural._classes.affine import Affine


np.random.seed(2)
random.seed(0)


@pytest.fixture
def model():
    model = Affine(2, 2)
    return model


def test_init(model):
    assert model.nO == 2
    assert model.nI == 2
    assert model.W is not None
    assert model.b is not None


def test_predict_bias(model):
    input_ = model.ops.allocate((1, model.nI))
    target_scores = model.ops.allocate((1, model.nO))
    scores = model(input_)
    assert_allclose(scores[0], target_scores[0])

    # Set bias for class 0
    model.b[0] = 2.0
    target_scores[0, 0] = 2.0
    scores = model(input_)
    assert_allclose(scores, target_scores)

    # Set bias for class 1
    model.b[1] = 5.0
    target_scores[0, 1] = 5.0
    scores = model(input_)
    assert_allclose(scores, target_scores)


@pytest.mark.parametrize(
    "X,expected",
    [
        (np.asarray([0.0, 0.0], dtype="f"), [0.0, 0.0]),
        (np.asarray([1.0, 0.0], dtype="f"), [1.0, 0.0]),
        (np.asarray([0.0, 1.0], dtype="f"), [0.0, 1.0]),
        (np.asarray([1.0, 1.0], dtype="f"), [1.0, 1.0]),
    ],
)
def test_predict_weights(X, expected):
    W = np.asarray([1.0, 0.0, 0.0, 1.0], dtype="f").reshape((2, 2))
    bias = np.asarray([0.0, 0.0], dtype="f")
    print(W)
    print(bias)

    model = Affine(W.shape[0], W.shape[1])
    model.W[:] = W
    model.b[:] = bias

    scores = model.predict_one(X)
    print("scores", scores, "expected", expected)
    assert_allclose(scores, expected)


def test_update():
    W = np.asarray([1.0, 0.0, 0.0, 1.0], dtype="f").reshape((2, 2))
    bias = np.asarray([0.0, 0.0], dtype="f")

    model = Affine(2, 2)
    model.W[:] = W
    model.b[:] = bias
    sgd = SGD(model.ops, 1.0, L2=0.0)
    sgd.averages = None

    ff = np.asarray([[0.0, 0.0]], dtype="f")
    tf = np.asarray([[1.0, 0.0]], dtype="f")
    ft = np.asarray([[0.0, 1.0]], dtype="f")  # noqa: F841
    tt = np.asarray([[1.0, 1.0]], dtype="f")  # noqa: F841

    # ff, i.e. 0, 0
    scores, finish_update = model.begin_update(ff)
    assert_allclose(scores[0, 0], scores[0, 1])
    # Tell it the answer was 'f'
    gradient = np.asarray([[-1.0, 0.0]], dtype="f")
    finish_update(gradient, sgd)

    assert model.b[0] == 1.0
    assert model.b[1] == 0.0
    # Unchanged -- input was zeros, so can't get gradient for weights.
    assert model.W[0, 0] == 1.0
    assert model.W[0, 1] == 0.0
    assert model.W[1, 0] == 0.0
    assert model.W[1, 1] == 1.0

    # tf, i.e. 1, 0
    scores, finish_update = model.begin_update(tf)
    # Tell it the answer was 'T'
    gradient = np.asarray([[0.0, -1.0]], dtype="f")
    finish_update(gradient, sgd)

    assert model.b[0] == 1.0
    assert model.b[1] == 1.0
    # Gradient for weights should have been outer(gradient, input)
    # so outer([0, -1.], [1., 0.])
    # =  [[0., 0.], [-1., 0.]]
    assert model.W[0, 0] == 1.0 - 0.0
    assert model.W[0, 1] == 0.0 - 0.0
    assert model.W[1, 0] == 0.0 - -1.0
    assert model.W[1, 1] == 1.0 - 0.0
