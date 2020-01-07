import pytest
from srsly import pickle_loads, pickle_dumps
from thinc.layers import with_list2array, Linear


@pytest.fixture
def linear():
    return Linear(5, 3)


def test_pickle_with_flatten(linear):
    Xs = [linear.ops.allocate((2, 3)), linear.ops.allocate((4, 3))]
    model = with_list2array(linear)
    pickled = pickle_dumps(model)
    loaded = pickle_loads(pickled)
    Ys = loaded.predict(Xs)
    assert len(Ys) == 2
    assert Ys[0].shape == (Xs[0].shape[0], linear.get_dim("nO"))
    assert Ys[1].shape == (Xs[1].shape[0], linear.get_dim("nO"))
