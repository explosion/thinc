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


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true",
        help="include slow tests")


def pytest_runtest_setup(item):
    for opt in ['slow']:
        if opt in item.keywords and not item.config.getoption("--%s" % opt):
            pytest.skip("need --%s option to run" % opt)
