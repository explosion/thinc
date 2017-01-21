import pytest
import numpy
from numpy.testing import assert_allclose
from cytoolz import concat
from mock import Mock


from ...neural._classes.window_encode import MaxoutWindowEncode
from ...neural._classes.embed import Embed
from ...neural.ops import NumpyOps


@pytest.fixture
def nr_out():
    return 5


@pytest.fixture
def ndim():
    return 3


@pytest.fixture
def total_length(positions):
    return sum(len(occurs) for occurs in positions.values())


@pytest.fixture
def ops():
    return NumpyOps()


@pytest.fixture
def nV(positions):
    return max(positions.keys()) + 1


@pytest.fixture
def model(ops, nr_out, ndim, nV):
    model = MaxoutWindowEncode(
                Embed(ndim, ndim, nV), nr_out, pieces=2, window=2)
    return model


@pytest.fixture
def B(positions):
    return sum(len(occurs) for occurs in positions.values())


@pytest.fixture
def gradients_BO(model, positions, B):
    gradient = model.ops.allocate((B, model.nO)) 
    for i in range(gradient.shape[0]):
        gradient[i] -= i
    return gradient


@pytest.fixture
def O(nr_out):
    return nr_out


@pytest.fixture
def I(ndim):
    return ndim


def test_nI(model):
    assert model.nI == model.embed.nO

def test_shape_is_inferred_from_data(positions, ndim, nV):
    model = MaxoutWindowEncode(
                Embed(ndim, ndim, nV), pieces=2, window=2)
    y = model.ops.asarray([[0, 1]], dtype='i')
    with model.begin_training(positions, y):
        pass
    assert model.nO == 2


def test_predict_matches_update(B, I, O, model, positions):
    x1 = model.predict(positions)
    x2, _ = model.begin_update(positions)
    assert_allclose(x1, x2)


def test_update_shape(B, I, O, model, positions, gradients_BO):
    assert gradients_BO.shape == (B, O)
    fwd, finish_update = model.begin_update(positions)
    null_grad = finish_update(gradients_BO, sgd=None)
    assert null_grad is None


def test_embed_fine_tune_is_called(model):
    positions = {10: [0]}
    W = model.W
    W.fill(1)
    gradients_BO = numpy.zeros((1, model.nO), dtype='f') - 1.
    mock_fine_tune = Mock()
    model.embed.begin_update = replace_finish_update(
        model.embed.begin_update, mock_fine_tune)
    fwd, finish_update = model.begin_update(positions)
    finish_update(gradients_BO, sgd=None)
    mock_fine_tune.assert_called_once()


def test_embed_static(model):
    positions = {10: [0]}
    W = model.W
    W.fill(1)
    gradients_BO = numpy.zeros((1, model.nO), dtype='f') - 1.
    model.embed.begin_update = replace_finish_update(
        model.embed.begin_update, None)
    fwd, finish_update = model.begin_update(positions)
    grad = finish_update(gradients_BO, sgd=None)
    assert grad is None


def replace_finish_update(begin_update, replacement):
    def replaced(*args, **kwargs):
        X, finish_update = begin_update(*args, **kwargs)
        return X, replacement
    return replaced


def test_weights_change_fine_tune(model):
    # Replace backward pass of Embed, so that it passes through the gradient.
    model.embed.begin_update = replace_finish_update(
            model.embed.begin_update, lambda gradient, sgd=None: gradient)
 
    positions = {10: [0]}
    model.W *= 0.
    model.b *= 0.
    gradients_BO = numpy.zeros((1, model.nO), dtype='f') - 1.
    
    fwd, finish_update = model.begin_update(positions)
    grad1 = finish_update(gradients_BO, sgd=None)
    
    model.W += 1
    fwd, finish_update = model.begin_update(positions)
    grad2 = finish_update(gradients_BO, sgd=None)
    
    for val1, val2 in zip(grad1.flatten(), grad2.flatten()):
        assert val1 != val2
