import pytest
import numpy

from ...ids2vecs import WindowEncode
from ...optimizers import SGD


class MyWindowEncode(WindowEncode):
    device = 'cpu'

    def add_param(self, id_, shape):
        pass


@pytest.fixture
def model():
    model = MyWindowEncode(device='cpu', nr_in=10, nr_out=3, vectors={})
    assert model.nr_out == 3
    return model


def test_init():
    model = MyWindowEncode(device='cpu', nr_out=10, nr_in=2, vectors={})
    assert model.nr_out == 10
    assert model.W.shape == (10, 3, 5, 2)


def test_get_ids(model):
    x = [[1, 10], [20, 20]]
    ids = model._get_positions(x)
    assert ids == {1: [(0, 0)], 10: [(0, 1)], 20: [(1, 0), (1, 1)]}


def test_dot_ids(model):
    x = [[1, 10], [20, 20]]
    ids = model._get_positions(x)
    assert ids == {1: [(0, 0)], 10: [(0, 1)], 20: [(1, 0), (1, 1)]}
    model._dot_ids(ids, [2, 2])


@pytest.mark.xfail
def test_predict_batch(model):
    x = [[1, 10], [20, 20]]
    scores = model.predict_batch(x)


@pytest.mark.xfail
def test_update(model):
    x = [[1, 10], [20, 20]]
    scores, update = model.begin_update(x)
    grads = [np.zeros((len(eg), model.nr_out)) for eg in x]
    update(grads)


@pytest.mark.xfail
def set_vector(model, key, values):
    model.add_vector(key, (len(values),))
    vector = model.get_vector(key)
    for i, value in enumerate(values):
        vector[i] = value
 

@pytest.mark.xfail
def test_predict_shapes():
    model = WindowEncode('encode', nr_out=5, nr_in=2)
    set_vector(model, 1, [1., -1.])
    set_vector(model, 2, [0.25, -0.25])

    output = model.predict_batch(numpy.asarray([[1], [2, 1]]))
    assert len(output) == 2
    assert output[0].shape == (1, 5)
    assert output[1].shape == (2, 5)


@pytest.mark.xfail
def test_update():
    model = WindowEncode(nr_out=2, nr_in=2, nr_piece=2)
    model.W.fill(2)
    model.b.fill(0) 
    set_vector(model, 2, [0.25, -0.25])
    
    output, finish_update = model.begin_update(numpy.asarray([[2]]))
    finish_update(numpy.asarray([[[0.25, 0.25]]]), SGD(model.ops, 1.0))
    vector = model.get_vector(2)
    # Not positive these are correct?
    assert vector[0] == -0.75
    assert vector[1] == -1.25
    assert list(model.b[0]) == [0.0, 0.0]
