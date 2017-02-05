import pytest
import numpy
from numpy.testing import assert_allclose

from ...neural._classes.model import Model
from ...neural.vecs2vec import Pooling, mean_pool, max_pool


@pytest.fixture(params=[[mean_pool], [max_pool], [mean_pool, max_pool]])
def model(request):
    return Pooling(*request.param)


@pytest.fixture
def X(nB, nI):
    return [numpy.zeros((nB, nI))+i for i in range(5)]


@pytest.fixture
def dY(X, nI):
    return numpy.ones((len(X), nI))


#def test_pools_are_created_successfully(model):
#    assert hasattr(model, 'predict')
#    assert hasattr(model, 'begin_update')
#    assert isinstance(model, Model)
#
#
#def test_pools_predict_shapes(model, X, nB, nI):
#    y = model.predict(X)
#    assert y.shape == (len(X), nI)
#
#
#def test_pools_begin_update_shapes(model, X, nB, nI):
#    y, _ = model.begin_update(X)
#    assert y.shape == (len(X), nI)


#def test_pools_finish_update_shapes(model, X, dY, nB, nI):
#    y, finish_update = model.begin_update(X)
#    gradient = finish_update(dY)
#    assert len(gradient) == len(X)
#    assert all([g.shape == x.shape for g, x in zip(gradient, X)])


#@pytest.xfail
#def test_pools_predict_matches_finish_update(model, X):
#    y = model.predict(X)
#    y2, _ = model.begin_update(X)
#    assert_allclose(y, y2)
#
#
#@pytest.xfail
#def test_zero_length_input_succeeds(model):
#    zero = numpy.ones((0, 10))
#    ones = numpy.ones((5, 10))
#    y = model.predict([zero, ones])
#    assert y.shape == (2, 10)
