import pytest

from ....neural._classes.window_encode import MaxoutWindowEncode
from ....neural._classes.embed import Embed
from ....neural.ops import NumpyOps

@pytest.fixture
def ndim():
    return 10

@pytest.fixture
def embed():
    return Embed(ndim)

@pytest.fixture
def model():
    return MaxoutWindowEncode(embed, 10)

def test_init_succeeds():
    model = MaxoutWindowEncode(embed, 10)


def test_init_defaults_and_overrides(model):
    model = MaxoutWindowEncode(Embed(10, 10), 10)
    assert model.nP == MaxoutWindowEncode.nP
    assert model.nW == MaxoutWindowEncode.nW
    assert model.nO == 10
    assert model.nI == None
    model = MaxoutWindowEncode(Embed(), 10, nP=MaxoutWindowEncode.nP-1)
    assert model.nP == MaxoutWindowEncode.nP-1
    model = MaxoutWindowEncode(10, nW=MaxoutWindowEncode.nF-1)
    assert model.nF == MaxoutWindowEncode.nF-1
