import pytest
from thinc.backends.mem import Memory
from thinc.api import NumpyOps


@pytest.fixture
def ops():
    return NumpyOps()


@pytest.mark.parametrize("size", [0, 10, 1000, 7, 12])
def test_init_allocates_mem(ops, size):
    params = Memory(ops, size)
    assert params._mem[0].size == size
    assert params._i == 0


@pytest.mark.parametrize("size", [-10, -1000, -1, -7, -12])
def test_init_rejects_negative_sizes(ops, size):
    with pytest.raises(ValueError):
        Memory(ops, size)


def test_add_param_within_size(ops):
    params = Memory(ops, size=128)
    params.add("W", (5, 10))
    assert params._offsets["W"] == (0, 0, (5, 10))
    params.add("b", (5,))
    assert params._offsets["b"] == (5 * 10, 0, (5,))


def test_add_param_realloc(ops):
    params = Memory(ops, size=10)
    params.add("b", (5,))
    assert params._offsets["b"] == (0, 0, (5,))
    params.add("W", (5, 10))
    assert params._offsets["W"] == (5, 0, (5, 10))
    assert params._offsets["b"] == (0, 0, (5,))


def test_get_param_present(ops):
    params = Memory(ops, size=10)
    b = params.add("b", (5,))
    b2 = params.get("b")
    b[0] = 100
    assert b[0] == b2[0]


def test_get_param_absent(ops):
    params = Memory(ops, size=10)
    b = params.get("b")
    assert b is None


def test_get_first_gradient(ops):
    params = Memory(ops, size=10)
    b = params.add("b", (5,))
    db = params.get("d_b")
    assert db is None
    params.add_gradient("d_b", "b")
    db = params.get("d_b")
    assert db.shape == b.shape


def test_get_existing_gradient(ops):
    params = Memory(ops, size=10)
    params.add("b", (5,))
    db = params.add_gradient("d_b", "b")
    db += 1
    db = params.get("d_b")
    assert db[0] == 1


def test_get_gradient_absent_parameter(ops):
    params = Memory(ops, size=10)
    d_b = params.get("d_b")
    assert d_b is None
