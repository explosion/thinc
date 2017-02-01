import pytest
from numpy.testing import assert_allclose
from collections import defaultdict


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


@pytest.fixture
def ids():
    return [1, 2, 3, 0, 0, 50, 6, 1, 0, 0, 4, 2, 9]


@pytest.fixture
def positions(ids):
    # [[1, 2, 3], [50, 6, 1], [4, 2, 9]]
    # {
    #   1: [(0, 0), (1, 2)],
    #   2: [(0, 1), (2, 1)],
    #   3: [(0,0)],
    #   50: [(1, 0)],
    #   6: [(1,1)],
    #   4: [(2,0)],
    #   9: [(2,2)]
    # }
    positions = defaultdict(list)
    for i, id_ in enumerate(ids):
        positions[id_].append(i)
    return positions


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true",
        help="include slow tests")


def pytest_runtest_setup(item):
    for opt in ['slow']:
        if opt in item.keywords and not item.config.getoption("--%s" % opt):
            pytest.skip("need --%s option to run" % opt)
