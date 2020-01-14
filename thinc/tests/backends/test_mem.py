import pytest
from thinc.backends.mem import Memory
from thinc.backends._param_server import ParamServer
from thinc.api import NumpyOps
import numpy


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
    assert "b" in params
    db = params.add_gradient("d_b", "b")
    db += 1
    db = params.get("d_b")
    assert db[0] == 1
    assert numpy.array_equal(params.weights, numpy.zeros((5,), dtype="f"))
    assert numpy.array_equal(params.gradient, numpy.ones((5,), dtype="f"))


def test_get_gradient_absent_parameter(ops):
    params = Memory(ops, size=10)
    d_b = params.get("d_b")
    assert d_b is None


def test_param_server_init():
    array = numpy.zeros((5,), dtype="f")
    params = {("a", 1): array, ("b", 2): array}
    grads = {("a", 1): array, ("c", 3): array}
    ps = ParamServer(params, grads)
    assert ps.param_keys == (("a", 1), ("b", 2))
    assert ps.grad_keys == (("a", 1),)
