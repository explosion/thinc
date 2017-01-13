import pytest
import numpy
from numpy.testing import assert_allclose

from ..._classes.feed_forward import FeedForward
from ..._classes.relu import Affine


@pytest.fixture
def dims():
    return [3, 4, 2]


@pytest.fixture
def nB():
    return 6

@pytest.fixture
def nI(dims):
    return dims[0]


@pytest.fixture
def nH(dims):
    return dims[1]


@pytest.fixture
def nO(dims):
    return dims[2]


@pytest.fixture
def model1(nH, nI):
    model = Affine(nH, nI)
    return round_weights(model)


@pytest.fixture
def model2(nO, nH):
    model = Affine(nO, nH)
    return round_weights(model)


@pytest.fixture
def input_data(nB, nI):
    return numpy.ones((nB, nI), dtype='f') + 1.


@pytest.fixture
def gradient_data(nB, nO):
    return numpy.zeros((nB, nO), dtype='f') -1.


@pytest.fixture
def model(model1, model2):
    return FeedForward(model1, model2)


def round_weights(model):
    # Clamp weights to integers, so that we don't get annoying float stuff.
    #model.W[:] = numpy.round(model.W)
    #model.b[:] = numpy.round(model.b)
    return model


def get_expected_predict(input_data, Ws, bs):
    X = input_data
    for (W, b) in zip(Ws, bs):
        X = numpy.ascontiguousarray(X)
        X = numpy.tensordot(X, W, axes=[[1], [1]]) + b
    return X


def numeric_gradient(predict, weights, epsilon=1e-4):
    out1 = predict(weights + epsilon)
    out2 = predict(weights - epsilon)
    return (out1 - out2) / (2 * epsilon)


def test_models_have_shape(model1, model2, nI, nH, nO):
    assert model1.W.shape == (nH, nI)
    assert model1.b.shape == (nH,)
    assert model2.W.shape == (nO, nH)
    assert model2.b.shape == (nO,)


def test_model_shape(model, model1, model2, nI, nH, nO):
    assert model.input_shape == model1.input_shape
    assert model.output_shape == model2.output_shape


def test_predict_and_begin_update_match(model, model1, model2, input_data):
    model = FeedForward(model1, model2)
    via_predict = model.predict(input_data)
    via_update, _ = model.begin_update(input_data)
    assert_allclose(via_predict, via_update)
    expected = get_expected_predict(input_data,
                [model1.W, model2.W],
                [model1.b, model2.b])
    assert_allclose(via_update, expected)
    assert expected.sum() != 0.


class GradientSpy(object):
    def __init__(self):
        self.weights = None
        self.d_weights = None
    def __call__(self, weights, grad):
        self.weights = weights
        self.d_weights = grad


def test_gradient(model1, input_data, nB, nH, nI, nO):
    model = model1
    truth = numpy.zeros((nB, nH), dtype='float64')
    
    guess, backprop = model.begin_update(input_data)
    backprop(guess - truth)
    analytic_gradient = model.mem.gradient.copy()

    def predict(i, update):
        model.mem.weights[i] += update
        X = model.predict(input_data)
        model.mem.weights[i] -= update
        return X

    numeric_gradient = get_numeric_gradient(predict, model.mem.weights.size, truth)
    # Set this for all parameters.
    assert_allclose(analytic_gradient, numeric_gradient, atol=0.01, rtol=1e-3)


def get_numeric_gradient(predict, n, target):
    gradient = numpy.zeros(n)
    for i in range(n):
        out1 = predict(i, 1e-4)
        out2 = predict(i, -1e-4)

        err1 = _get_loss(out1, target)
        err2 = _get_loss(out2, target)
        gradient[i] = (err1 - err2) / (2 * 1e-4)
    return gradient

    
def _get_loss(truth, guess): 
    return numpy.sum(numpy.sum(0.5*numpy.square(truth - guess),1))
