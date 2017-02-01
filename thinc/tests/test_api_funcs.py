import pytest
import numpy
from numpy.testing import assert_allclose

from ..api import chain, layerize, metalayerize, noop, clone
from ..neural._classes.affine import Affine
from ..neural._classes.model import Model


@pytest.fixture
def model1(nH, nI):
    return Affine(nH, nI)


@pytest.fixture
def model2(nO, nH):
    return Affine(nO, nH)


@pytest.fixture
def model3(nO):
    return Affine(nO, nO)


def test_chain_zero():
    model = chain()
    assert isinstance(model, Model)

def test_chain_one(model1):
    model = chain(model1)
    assert isinstance(model, Model)


def test_chain_two(model1, model2):
    model = chain(model1, model2)
    assert len(model._layers) == 2


def test_chain_right_branch(model1, model2, model3):
    merge1 = chain(model1, model2)
    merge2 = chain(merge1, model3)
    assert len(merge2._layers) == 3


def test_clone(model1, nI):
    ones = numpy.ones((10, nI))
    model1.nI = None
    model = clone(model1, 10)
    model.begin_training(ones)
    output_from_cloned = model(ones)
    output_from_orig = model1(ones)
    assert output_from_cloned.sum() != output_from_orig.sum()


def test_layerize_predict_noop(model1, model2, nI):
    ones = numpy.ones((10, nI))
    model = layerize(noop(model1, model2))  
    y = model(ones)
    assert_allclose(y, ones)


def test_layerize_update_noop(model1, model2, nI):
    ones = numpy.ones((10, nI))
    model = layerize(noop(model1, model2))
    y, finish_update = model.begin_update(ones)
    assert_allclose(y, ones)
    grad_in = numpy.ones(y.shape) + 1.
    grad_out = finish_update(grad_in)
    assert_allclose(grad_in, grad_out)


def test_layerize_prespecify_predict(model1, model2, nI):
    def noop_predict(X):
        return X
    @layerize(predict=noop_predict, predict_one=noop_predict)
    def noop_model(X, drop=0.):
        return X, lambda d, sgd=None: d
    ones = numpy.ones((10, nI))
    assert_allclose(ones, noop_model.predict(ones))
    assert_allclose(ones[0], noop_model.predict_one(ones[0]))


def test_metalayerize_noop(model1, model2, nI):
    @metalayerize
    def meta_noop(layers, X, *args, **kwargs):
        def finish_update(d, sgd=None):
            return d
        return X, finish_update
    ones = numpy.ones((10, nI))
    model = meta_noop((model1, model2))
    
    y = model(ones)
    assert_allclose(y, ones)
    
    y, finish_update = model.begin_update(ones)
    assert_allclose(y, ones)
    grad_in = numpy.ones(y.shape) + 1.
    grad_out = finish_update(grad_in)
    assert_allclose(grad_in, grad_out)


def test_chain_dimension_mismatch(model1, model2):
    pass


def test_chain_begin_update(model1, model2):
    pass


def test_chain_learns(model1, model2):
    pass
