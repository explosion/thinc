import pytest
import numpy
from thinc.api import get_width, Ragged, Padded, minibatch
from thinc.util import get_array_module, is_numpy_array


@pytest.mark.parametrize(
    "obj,width",
    [
        (numpy.zeros((1, 2, 3, 4)), 4),
        (numpy.array(1), 0),
        (numpy.array([1, 2]), 3),
        ([numpy.zeros((1, 2)), numpy.zeros((1))], 2),
        (Ragged(numpy.zeros((1, 2)), numpy.zeros(1)), 2),
        (Padded(numpy.zeros((1, 2)), numpy.zeros(1)), 2),
        ([], 0),
    ],
)
def test_get_width(obj, width):
    assert get_width(obj) == width


@pytest.mark.parametrize(
    "obj", [1234, "foo", {"a": numpy.array(0)}],
)
def test_get_width_fail(obj):
    with pytest.raises(ValueError):
        get_width(obj)


def test_array_module_cpu_gpu_helpers():
    xp = get_array_module(0)
    assert hasattr(xp, "ndarray")
    assert is_numpy_array(numpy.zeros((1, 2)))
    assert not is_numpy_array((1, 2))


def test_minibatch():
    items = [1, 2, 3, 4, 5, 6]
    batches = minibatch(items, 3)
    assert list(batches) == [[1, 2, 3], [4, 5, 6]]
    batches = minibatch(items, (i for i in (3, 2, 1)))
    assert list(batches) == [[1, 2, 3], [4, 5], [6]]
    items = (i for i in range(1, 7))
    batches = minibatch(items, 3)
    assert list(batches) == [[1, 2, 3], [4, 5, 6]]
    items = (i for i in range(1, 7))
    batches = minibatch(items, (i for i in (3, 2, 1, 1)))
    assert list(batches) == [[1, 2, 3], [4, 5], [6]]
