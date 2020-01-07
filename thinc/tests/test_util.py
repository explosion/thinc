import pytest
import numpy
from thinc.api import get_width, Ragged, Padded


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
