import pytest
from mock import MagicMock, Mock
from hypothesis import given, settings
import numpy
from numpy.testing import assert_allclose
from thinc.neural._classes.affine import Affine
from thinc.neural.ops import NumpyOps

from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape


@pytest.fixture
def model():
    model = Affine()
    return model


def test_Affine_default_name(model):
    assert model.name == "affine"


def test_Affine_calls_default_descriptions():
    orig_desc = dict(Affine.descriptions)
    Affine.descriptions = {
        name: Mock(desc) for (name, desc) in Affine.descriptions.items()
    }
    model = Affine()

    assert len(model.descriptions) == 6
    for name, desc in model.descriptions.items():
        desc.assert_called()
    assert "nI" in model.descriptions
    assert "nO" in model.descriptions
    assert "W" in model.descriptions
    assert "b" in model.descriptions
    assert "d_W" in model.descriptions
    assert "d_b" in model.descriptions
    Affine.descriptions = orig_desc


def test_Affine_dimensions_on_data():
    X = MagicMock(shape=(5, 10))
    y = MagicMock(shape=(8,))
    y.max = MagicMock()
    model = Affine()
    model.on_data_hooks = model.on_data_hooks[:1]
    model.begin_training(X, y)
    assert model.nI is not None
    y.max.assert_called_with()


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_begin_update_matches_predict(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    fwd_via_begin_update, finish_update = model.begin_update(input_)
    fwd_via_predict_batch = model.predict(input_)
    assert_allclose(fwd_via_begin_update, fwd_via_predict_batch)


@pytest.mark.skip
@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_dropout_gives_zero_activations(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    fwd_dropped, _ = model.begin_update(input_)
    assert all(val == 0.0 for val in fwd_dropped.flatten())


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_dropout_gives_zero_gradients(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    fwd_dropped, finish_update = model.begin_update(input_, drop=1.0)
    grad_BO = numpy.ones((nr_batch, nr_out), dtype="f")
    grad_BI = finish_update(grad_BO)
    assert all(val == 0.0 for val in grad_BI.flatten())


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
    grad_BI = finish_update(grad_BO, sgd)  # noqa: F841
    assert seen_keys == {model.id}


@pytest.mark.xfail
def test_begin_update_not_batch():
    model = Affine(4, 5)
    input_ = model.ops.allocate((6,))
    with pytest.raises(TypeError):
        model.begin_update(input_)


@pytest.mark.skip
def test_predict_update_dim_mismatch():
    model = Affine(4, 5, ops=NumpyOps())
    input_ = model.ops.allocate((10, 9))
    with pytest.raises(TypeError):
        model.begin_update(input_)


@settings(max_examples=100)
@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_predict_small(W_b_input):
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = Affine(nr_out, nr_in)
    model.W[:] = W
    model.b[:] = b

    einsummed = numpy.einsum(
        "oi,bi->bo",
        numpy.asarray(W, dtype="float64"),
        numpy.asarray(input_, dtype="float64"),
        optimize=False,
    )

    expected_output = einsummed + b

    predicted_output = model.predict(input_)
    assert_allclose(predicted_output, expected_output, rtol=0.01, atol=0.01)


@pytest.mark.skip
@given(arrays_OI_O_BI(max_batch=100, max_out=100, max_in=100))
def test_predict_extensive(W_b_input):
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = Affine(nr_out, nr_in)
    model.W[:] = W
    model.b[:] = b

    einsummed = numpy.einsum(
        "oi,bi->bo",
        numpy.asarray(W, dtype="float64"),
        numpy.asarray(input_, dtype="float64"),
        optimize=False,
    )

    expected_output = einsummed + b

    predicted_output = model.predict(input_)
    assert_allclose(predicted_output, expected_output, rtol=1e-04, atol=0.0001)
