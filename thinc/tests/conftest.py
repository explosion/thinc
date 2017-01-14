import pytest
from numpy.testing import assert_allclose


@pytest.fixture(params=[1, 2, 9])
def nB(request):
    return request.param

@pytest.fixture(params=[1, 6])
def nI(request):
    return request.param


@pytest.fixture(params=[1, 5, 3])
def nH(request):
    return request.param


@pytest.fixture(params=[1, 2, 7, 9])
def nO(request):
    return request.param
