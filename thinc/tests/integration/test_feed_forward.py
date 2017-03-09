import pytest
import numpy
from numpy.testing import assert_allclose

from ...neural._classes.feed_forward import FeedForward
from ...neural._classes.affine import Affine
from ...neural._classes.relu import ReLu
from ...neural._classes.softmax import Softmax

@pytest.fixture
def model1(nH, nI):
    model = ReLu(nH, nI)
    return model


@pytest.fixture
def model2(nO, nH):
    model = Affine(nO, nH)
    return model


@pytest.fixture
def input_data(nB, nI):
    return numpy.ones((nB, nI), dtype='f') + 1.


@pytest.fixture
def gradient_data(nB, nO):
    return numpy.zeros((nB, nO), dtype='f') -1.


@pytest.fixture
def model(model1, model2):
    return FeedForward((model1, model2))


def get_expected_predict(input_data, Ws, bs):
    X = input_data
    for i, (W, b) in enumerate(zip(Ws, bs)):
        X = numpy.ascontiguousarray(X)
        if i > 0:
            X *= X > 0
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
    model = FeedForward((model1, model2))
    via_predict = model.predict(input_data)
    via_update, _ = model.begin_update(input_data)
    assert_allclose(via_predict, via_update)
    expected = get_expected_predict(input_data,
                [model1.W, model2.W],
                [model1.b, model2.b])
    assert_allclose(via_update, expected, atol=1e-2, rtol=1e-4)


class GradientSpy(object):
    def __init__(self):
        self.weights = None
        self.d_weights = None
    def __call__(self, weights, grad):
        self.weights = weights
        self.d_weights = grad


def test_gradient(model, input_data, nB, nH, nI, nO):
    truth = numpy.zeros((nB, nO), dtype='float32')
    truth[0] = 1.0
    
    guess, backprop = model.begin_update(input_data)
    backprop(guess - truth)

    for layer in model._layers:
        def predict(i, update):
            layer._mem.weights[i] += update
            X = model.predict(input_data)
            layer._mem.weights[i] -= update
            return X
        agrad = layer._mem.gradient.copy()
        ngrad = get_numeric_gradient(predict, layer._mem.weights.size, truth)
        assert_allclose(agrad, ngrad, atol=0.2, rtol=0.2)


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
