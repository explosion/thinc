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


def set_vector(model, key, values):
    model.add_param(key, (len(values),))
    vector = model.get_param(key)
    for i, value in enumerate(values):
        vector[i] = value
 

@pytest.mark.xfail
def test_init():
    model = WindowEncode('encode', (10,), (5,), tuple())
    assert model.nr_out == 10


@pytest.mark.xfail
def test_add_param():
    model = WindowEncode('encode', (2,), (2,), [])
    model.add_param(1, (2,))
    vector = model.get_param(1)
    assert vector.shape == (2,)
    assert vector.sum() != 0


@pytest.mark.xfail
def test_predict_shapes():
    model = WindowEncode('encode', (5,), (2,), [])
    set_vector(model, 1, [1., -1.])
    set_vector(model, 2, [0.25, -0.25])

    output = model.predict_batch(numpy.asarray([[1], [2, 1]]))
    assert len(output) == 2
    assert output[0].shape == (1, 5)
    assert output[1].shape == (2, 5)


@pytest.mark.xfail
def test_update():
    for side in [0, 1]:
        model = WindowEncode('encode', (2,), (2,), [])
        model.W.fill(1)
        model.b.fill(0)
        if side == 0:
            model.b[0, 0] = 1.0
            model.b[1, 0] = 0.25
        else:
            model.b[0, 1] = 0.25
            set_vector(model, 1, [1., -1.])
        set_vector(model, 2, [0.25, -0.25])
        model.is_initialized = True

        output, finish_update = model.begin_update(numpy.asarray([[2]]))
        finish_update(numpy.asarray([[[0.25, 0.25]]]), SGD(1.0))
        vector = model.get_param(2)
        assert vector[0] == -0.25
        assert vector[1] == -0.75
        if side == 0:
            assert list(model.b[0]) == [0.75, 0.0]
            assert list(model.b[1]) == [0.0, 0.0]
        else:
            assert list(model.b[0]) == [0.0, 0.0]
            assert list(model.b[1]) == [0.0, 0.75]
