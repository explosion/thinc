import pytest

from ....params import Params
from ....ops import NumpyOps

@pytest.fixture
def ops():
    return NumpyOps()

@pytest.mark.parametrize('size', [0, 10, 1000, 7, 12])
def test_init_allocates_mem(ops, size):
    params = Params(ops, size)
    assert params._mem.size == size
    assert params._i == 0

@pytest.mark.parametrize('size', [-10, -1000, -1, -7, -12])
def test_init_rejects_negative_sizes(ops, size):
    with pytest.raises(ValueError):
        params = Params(ops, size)

def test_add_param_within_size(ops):
    model = Params(ops, size=128)
    model.add('W', (5, 10))
    assert model._offsets['W'] == (0, (5, 10))
    model.add('b', (5,))
    assert model._offsets['b'] == (5*10, (5,))
    

def test_add_param_realloc(ops):
    model = Params(ops, size=10)
    model.add('b', (5,))
    assert model._offsets['b'] == (0, (5,))
    model.add('W', (5, 10))
    assert model._offsets['W'] == (5, (5, 10))
    assert model._offsets['b'] == (0, (5,))
 

def test_get_param_present(ops):
    model = Params(ops, size=10)
    b = model.add('b', (5,))
    b2 = model.get('b')
    b[0] = 100
    assert b[0] == b2[0]
 

def test_get_param_absent(ops):
    model = Params(ops, size=10)
    b = model.get('b')
    assert b is None
 

def test_get_first_gradient(ops):
    model = Params(ops, size=10)
    b = model.add('b', (5,))
    b2 = model.get('d_b')
    b[0] = 100
    assert b2[0] == 0
 

def test_get_existing_gradient(ops):
    model = Params(ops, size=10)
    b = model.add('b', (5,))
    b2 = model.get('d_b')
    b[0] = 100
    assert b2[0] == 0
    b2[0] = 20.
    b3 = model.get('d_b')
    assert b3[0] == b2[0]
 

def test_get_gradient_absent_parameter(ops):
    model = Params(ops, size=10)
    d_b = model.get('d_b')
    assert d_b is None
