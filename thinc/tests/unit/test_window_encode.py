import pytest
from ...ids2vecs import WindowEncode

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
    assert model.W.shape == (10, 3, 2)


@pytest.mark.xfail
def test_get_ids(model):
    x = [[1, 10], [20, 20]]
    ids = model._get_ids(x)
    assert ids == {1: [(0, 0)], 10: [(0, 1)], 20: [(1, 0), (1, 1)]}


@pytest.mark.xfail
def test_dot_ids(model):
    x = [[1, 10], [20, 20]]
    ids = model._get_ids(x)
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

