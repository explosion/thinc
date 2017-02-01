import pytest

from ...neural.mem import Memory
from ...neural.ops import NumpyOps

@pytest.fixture
def ops():
    return NumpyOps()

@pytest.mark.parametrize('size', [0, 10, 1000, 7, 12])
def test_init_allocates_mem(ops, size):
    params = Memory(ops, size)
    assert params._mem[0].size == size
    assert params._i == 0

@pytest.mark.parametrize('size', [-10, -1000, -1, -7, -12])
def test_init_rejects_negative_sizes(ops, size):
    with pytest.raises(ValueError):
        params = Memory(ops, size)

def test_add_param_within_size(ops):
    params = Memory(ops, size=128)
    params.add('W', (5, 10))
    assert params._offsets['W'] == (0, 0, (5, 10))
    params.add('b', (5,))
    assert params._offsets['b'] == (5*10, 0, (5,))
    

def test_add_param_realloc(ops):
    params = Memory(ops, size=10)
    params.add('b', (5,))
    assert params._offsets['b'] == (0, 0, (5,))
    params.add('W', (5, 10))
    assert params._offsets['W'] == (5, 0, (5, 10))
    assert params._offsets['b'] == (0, 0, (5,))
 

def test_get_param_present(ops):
    params = Memory(ops, size=10)
    b = params.add('b', (5,))
    b2 = params.get('b')
    b[0] = 100
    assert b[0] == b2[0]
 

def test_get_param_absent(ops):
    params = Memory(ops, size=10)
    b = params.get('b')
    assert b is None
 

@pytest.mark.xfail
def test_get_first_gradient(ops):
    params = Memory(ops, size=10)
    b = params.add('b', (5,))
    b2 = params.get('d_b')
    b[0] = 100
    assert b2[0] == 0
 

@pytest.mark.xfail
def test_get_existing_gradient(ops):
    params = Memory(ops, size=10)
    b = params.add('b', (5,))
    b2 = params.get('d_b')
    b[0] = 100
    assert b2[0] == 0
    b2[0] = 20.
    b3 = params.get('d_b')
    assert b3[0] == b2[0]
 

def test_get_gradient_absent_parameter(ops):
    params = Memory(ops, size=10)
    d_b = params.get('d_b')
    assert d_b is None


#def test_merge_empty_others(ops):
#    params = Memory(ops, size=10)
#    assert params.allow_resize
#    params.merge_params([])
#    assert params.allow_resize
#
#
#def test_merge_no_resize(ops):
#    parent = Memory(ops, size=5)
#    assert parent.allow_resize
#    child = Memory(ops, size=2)
#    w_parent = parent.add('W', (4,))
#    w_child = child.add('W', (2,))
#    child._mem[0, 0] = 10.0
#    assert parent._i == 4
#    assert child._i == 2
#    parent.merge_params([child])
#    assert not parent.allow_resize
#    assert not child.allow_resize
#    assert parent._i == 6
#    assert child._i == 2
#
#
#def test_merge_with_resize(ops):
#    parent = Memory(ops, size=5)
#    child = Memory(ops, size=5)
#    w_parent = parent.add('W', (4,))
#    w_parent[0] += 2.
#    w_child = child.add('W', (3,))
#    w_child[0] += 5.
#    parent.merge_params([child])
#    w_parent = parent.get('W')
#    w_child = child.get('W')
#    assert w_child[0] == 5
#    assert w_parent[0] == 2
#
#
#def test_resize_disallowed_after_merge(ops):
#    parent = Memory(ops, size=5)
#    child = Memory(ops, size=5)
#    w_parent = parent.add('W', (4,))
#    w_child = child.add('W', (3,))
#    parent.merge_params([child])
#    with pytest.raises(ValueError):
#        child.add('b', (2,))
#
#def test_merge_disallowed_after_merge(ops):
#    parent = Memory(ops, size=5)
#    child = Memory(ops, size=5)
#    parent.add('W', (4,))
#    child.add('W', (3,))
#    parent.merge_params([child])
#    parent2 = Memory(ops, size=5)
#    with pytest.raises(ValueError):
#        parent2.merge_params([parent])
#    with pytest.raises(ValueError):
#        parent.replace_mem([parent2])
