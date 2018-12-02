# coding: utf8
from __future__ import unicode_literals

import pytest
from mock import MagicMock
import numpy
import numpy.random
from numpy.testing import assert_allclose
from hypothesis import given

from ...neural._classes.batchnorm import BatchNorm
from ...api import layerize, noop
from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape


@pytest.fixture
def shape():
    return (10, 20)


@pytest.fixture
def layer(shape):
    dummy = layerize(noop())
    dummy.nO = shape[-1]
    return dummy


def test_batch_norm_init(layer):
    layer = BatchNorm(layer)


def test_batch_norm_weights_init_to_one(layer):
    layer = BatchNorm(layer)
    assert layer.G is not None
    assert all(weight == 1.0 for weight in layer.G.flatten())


def test_batch_norm_runs_child_hooks(layer):
    mock_hook = MagicMock()
    layer.on_data_hooks.append(mock_hook)
    layer = BatchNorm(layer)
    for hook in layer.on_data_hooks:
        hook(layer, None)
    mock_hook.assert_called()


def test_batch_norm_predict_maintains_shape(layer, shape):
    input_ = numpy.ones(shape)
    input1 = layer.predict(input_)
    assert_allclose(input1, input_)
    layer = BatchNorm(layer)
    output = layer.predict(input_)
    assert output.shape == input_.shape


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_finish_update_calls_optimizer_with_weights(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input

    model = BatchNorm(model)

    output, finish_update = model.begin_update(input_)

    seen_keys = set()

    def sgd(data, gradient, key=None, **kwargs):
        seen_keys.add(key)
        assert data.shape == gradient.shape
        assert data.ndim == 1
        assert gradient.ndim == 1

    grad_BO = numpy.ones((nr_batch, nr_out), dtype="f")
    grad_BI = finish_update(grad_BO, sgd)  # noqa: F841
    assert seen_keys == {model.id, model.child.id}
