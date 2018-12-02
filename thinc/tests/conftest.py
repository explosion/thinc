# coding: utf8
from __future__ import unicode_literals

import pytest
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
    parser.addoption("--slow", action="store_true", help="include slow tests")


def pytest_runtest_setup(item):
    def getopt(opt):
        # When using 'pytest --pyargs thinc' to test an installed copy of
        # thinc, pytest skips running our pytest_addoption() hook. Later, when
        # we call getoption(), pytest raises an error, because it doesn't
        # recognize the option we're asking about. To avoid this, we need to
        # pass a default value. We default to False, i.e., we act like all the
        # options weren't given.
        return item.config.getoption("--%s" % opt, False)

    for opt in ["slow"]:
        if opt in item.keywords and not getopt("--%s" % opt):
            pytest.skip("need --%s option to run" % opt)
