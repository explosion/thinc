import pytest
from hypothesis import given
import numpy
from numpy.testing import assert_allclose
from ....ops import NumpyOps
from ....vec2vec import Affine
from ....exceptions import ShapeError


from ...strategies import arrays_OI_O_BI

def get_model(W_b_input):
    ops = NumpyOps()
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = Affine(nr_out, nr_in, ops=ops)
    model.initialize_params(add_gradient=True)
    model.W[:] = W
    model.b[:] = b
    return model

def get_shape(W_b_input):
    W, b, input_ = W_b_input
    return input_.shape[0], W.shape[0], W.shape[1]
    

@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_begin_update_matches_predict_batch(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    fwd_via_begin_update, finish_update = model.begin_update(input_)
    fwd_via_predict_batch = model.predict_batch(input_)
    assert_allclose(fwd_via_begin_update, fwd_via_predict_batch)


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_dropout_gives_zero_activations(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    fwd_dropped, _ = model.begin_update(input_, dropout=1.0)
    assert all(val == 0. for val in fwd_dropped.flatten())


@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_dropout_gives_zero_gradients(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    fwd_dropped, finish_update = model.begin_update(input_, dropout=1.0)
    grad_BO = numpy.ones((nr_batch, nr_out))
    grad_BI = finish_update(grad_BO)
    assert all(val == 0. for val in grad_BI.flatten())


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
        assert model.params._i == (nr_out * nr_in) + nr_out
        assert data.shape[0] == (nr_out * nr_in) + nr_out, data.shape[0]

    grad_BO = numpy.ones((nr_batch, nr_out))
    grad_BI = finish_update(grad_BO, optimizer=sgd)
    assert seen_keys == {('', model.name)}


def test_predict_batch_not_batch():
    model = Affine(4, 5, ops=NumpyOps())
    input_ = model.ops.allocate((6,))
    with pytest.raises(ShapeError):
        model.begin_update(input_)


def test_predict_update_dim_mismatch():
    model = Affine(4, 5, ops=NumpyOps())
    input_ = model.ops.allocate((10, 9))
    with pytest.raises(ShapeError):
        model.begin_update(input_)
