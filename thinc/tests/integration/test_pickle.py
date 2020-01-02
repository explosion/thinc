import pytest
from srsly import pickle_loads, pickle_dumps
from thinc.layers.with_flatten import with_flatten
from thinc.layers.affine import Affine


@pytest.fixture
def affine():
    return Affine(5, 3)


def test_pickle_with_flatten(affine):
    Xs = [affine.ops.allocate((2, 3)), affine.ops.allocate((4, 3))]
    model = with_flatten(affine)
    pickled = pickle_dumps(model)
    loaded = pickle_loads(pickled)
    Ys = loaded.predict(Xs)
    assert len(Ys) == 2
    assert Ys[0].shape == (Xs[0].shape[0], affine.get_dim("nO"))
    assert Ys[1].shape == (Xs[1].shape[0], affine.get_dim("nO"))
