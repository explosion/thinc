from ...._classes.window_encode import MaxoutWindowEncode
from ...._classes.window_encode import _get_positions
from ....ops import NumpyOps

import pytest
import numpy


@pytest.fixture
def ids():
    return [[0,1,2], [3], [0,5,3,6]]


@pytest.fixture
def vector_length():
    return 3


@pytest.fixture
def vectors(ids, vector_length):
    max_id = max(max(seq) for seq in ids)
    vecs = numpy.zeros((max_id+1, 2))
    for i in range(max_id+1):
        vecs[i] += i
    return vecs


@pytest.fixture
def lengths(ids):
    return [len(seq) for seq in ids]


@pytest.fixture
def positions(ids):
    return _get_positions(ids)


def test_forward_succeeds(ids, positions, vectors, lengths):
    model = MaxoutWindowEncode(8, nr_in=2, ops=NumpyOps())
    model.initialize_params()
    out, whiches = model._forward(positions, vectors, lengths)
