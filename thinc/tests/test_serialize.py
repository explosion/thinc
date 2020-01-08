import pytest
import srsly
from thinc.api import with_list2array, Linear, Maxout, chain


@pytest.fixture
def linear():
    return Linear(5, 3)


def test_pickle_with_flatten(linear):
    Xs = [linear.ops.alloc_f2d(2, 3), linear.ops.alloc_f2d(4, 3)]
    model = with_list2array(linear)
    pickled = srsly.pickle_dumps(model)
    loaded = srsly.pickle_loads(pickled)
    Ys = loaded.predict(Xs)
    assert len(Ys) == 2
    assert Ys[0].shape == (Xs[0].shape[0], linear.get_dim("nO"))
    assert Ys[1].shape == (Xs[1].shape[0], linear.get_dim("nO"))


def test_simple_model_roundtrip_bytes():
    model = Maxout(5, 10, nP=2)
    b = model.get_param("b")
    b += 1
    data = model.to_bytes()
    b = model.get_param("b")
    b -= 1
    model = model.from_bytes(data)
    assert model.get_param("b")[0, 0] == 1


def test_multi_model_roundtrip_bytes():
    model = chain(Maxout(5, 10, nP=2), Maxout(2, 3))
    b = model.layers[0].get_param("b")
    b += 1
    b = model.layers[1].get_param("b")
    b += 2
    data = model.to_bytes()
    b = model.layers[0].get_param("b")
    b -= 1
    b = model.layers[1].get_param("b")
    b -= 2
    model = model.from_bytes(data)
    assert model.layers[0].get_param("b")[0, 0] == 1
    assert model.layers[1].get_param("b")[0, 0] == 2


def test_multi_model_load_missing_dims():
    model = chain(Maxout(5, 10, nP=2), Maxout(2, 3))
    b = model.layers[0].get_param("b")
    b += 1
    b = model.layers[1].get_param("b")
    b += 2
    data = model.to_bytes()

    model2 = chain(Maxout(5), Maxout())
    model2 = model2.from_bytes(data)
    assert model2.layers[0].get_param("b")[0, 0] == 1
    assert model2.layers[1].get_param("b")[0, 0] == 2
