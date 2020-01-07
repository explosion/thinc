import pytest
from mock import MagicMock
from hypothesis import given, settings
import numpy
from numpy.testing import assert_allclose
from thinc.layers import Linear, chain, Dropout

from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape


@pytest.fixture
def model():
    model = Linear()
    return model


def test_linear_default_name(model):
    assert model.name == "linear"


def test_linear_dimensions_on_data():
    X = MagicMock(shape=(5, 10))
    y = MagicMock(shape=(8,))
    y.max = MagicMock()
    model = Linear()
    model.initialize(X, y)
    assert model.get_dim("nI") is not None
    y.max.assert_called_with()


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_begin_update_matches_predict(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    fwd_via_begin_update, finish_update = model.begin_update(input_)
    fwd_via_predict_batch = model.predict(input_)
    assert_allclose(fwd_via_begin_update, fwd_via_predict_batch)


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_finish_update_calls_optimizer_with_weights(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    output, finish_update = model.begin_update(input_)

    seen_keys = set()

    def sgd(data, gradient, key=None, **kwargs):
        seen_keys.add(key)
        assert data.shape == gradient.shape
        assert data.ndim == 1
        assert gradient.ndim == 1
        assert model._mem._i == (nr_out * nr_in) + nr_out
        assert data.shape[0] == (nr_out * nr_in) + nr_out, data.shape[0]

    grad_BO = numpy.ones((nr_batch, nr_out), dtype="f")
    grad_BI = finish_update(grad_BO)  # noqa: F841
    model.finish_update(sgd)
    assert seen_keys == {model.id}


@settings(max_examples=100)
@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_predict_small(W_b_input):
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = Linear(nr_out, nr_in)
    model.set_param("W", W)
    model.set_param("b", b)

    einsummed = numpy.einsum(
        "oi,bi->bo",
        numpy.asarray(W, dtype="float64"),
        numpy.asarray(input_, dtype="float64"),
        optimize=False,
    )

    expected_output = einsummed + b

    predicted_output = model.predict(input_)
    assert_allclose(predicted_output, expected_output, rtol=0.01, atol=0.01)


@given(arrays_OI_O_BI(max_batch=20, max_out=30, max_in=30))
def test_predict_extensive(W_b_input):
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = Linear(nr_out, nr_in)
    model.set_param("W", W)
    model.set_param("b", b)

    einsummed = numpy.einsum(
        "bi,oi->bo",
        numpy.asarray(input_, dtype="float32"),
        numpy.asarray(W, dtype="float32"),
        optimize=False,
    )

    expected_output = einsummed + b

    predicted_output = model.predict(input_)
    assert_allclose(predicted_output, expected_output, rtol=1e-04, atol=0.0001)


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_dropout_gives_zero_activations(W_b_input):
    model = chain(get_model(W_b_input), Dropout(1.0))
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    fwd_dropped, _ = model.begin_update(input_)
    assert all(val == 0.0 for val in fwd_dropped.flatten())


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_dropout_gives_zero_gradients(W_b_input):
    model = chain(get_model(W_b_input), Dropout(1.0))
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    for node in model.walk():
        if node.name == "dropout":
            node.set_attr("rate", 1.0)
    fwd_dropped, finish_update = model.begin_update(input_)
    grad_BO = numpy.ones((nr_batch, nr_out), dtype="f")
    grad_BI = finish_update(grad_BO)
    assert all(val == 0.0 for val in grad_BI.flatten())
