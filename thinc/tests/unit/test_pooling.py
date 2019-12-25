import pytest
import numpy
from thinc.t2v import Pooling, mean_pool, max_pool


@pytest.fixture(params=[[mean_pool], [max_pool], [mean_pool, max_pool]])
def model(request):
    return Pooling(*request.param)


@pytest.fixture
def X(nB, nI):
    return [numpy.zeros((nB, nI)) + i for i in range(5)]


@pytest.fixture
def dY(X, nI):
    return numpy.ones((len(X), nI))
