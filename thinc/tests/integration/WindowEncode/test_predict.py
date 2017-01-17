from ....neural._classes.window_encode import MaxoutWindowEncode
from ....neural._classes.window_encode import _get_positions
from ....neural.ops import NumpyOps

import pytest
import numpy


try:
    import cytoolz as toolz
except ImportError:
    import toolz


@pytest.fixture
def ids():
    return [[0,1,2], [3], [0,5,3,6]]


@pytest.fixture
def vector_length():
    return 3


@pytest.fixture
def vectors(ids, vector_length):
    ids = list(toolz.concat(ids))
    vecs = numpy.zeros((len(ids), 2))
    for i, id_ in enumerate(ids):
        vecs[i] += i
    return vecs


@pytest.fixture
def lengths(ids):
    return [len(seq) for seq in ids]


@pytest.fixture
def positions(ids):
    return _get_positions(list(toolz.concat(ids)))

@pytest.mark.xfail
@pytest.fixture
def model(vectors):
    model = MaxoutWindowEncode(8, nr_in=len(vectors[0]), ops=NumpyOps())
    model.initialize_params()
    return model


