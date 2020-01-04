import numpy
from pydantic import create_model, ValidationError
from thinc.types import Floats1d, Floats2d
import pytest


@pytest.mark.parametrize(
    "arr,arr_type",
    [
        (numpy.zeros(0, dtype=numpy.float32), Floats1d),
        (numpy.zeros((1, 1), dtype=numpy.float32), Floats2d),
    ],
)
def test_array_validation_valid(arr, arr_type):
    test_model = create_model("TestModel", arr=(arr_type, ...))
    result = test_model(arr=arr)
    assert numpy.array_equal(arr, result.arr)


@pytest.mark.parametrize(
    "arr,arr_type",
    [
        (numpy.zeros(0, dtype=numpy.float64), Floats1d),
        (numpy.zeros((1, 1), dtype=numpy.float32), Floats1d),
    ],
)
def test_array_validation_invalid(arr, arr_type):
    test_model = create_model("TestModel", arr=(arr_type, ...))
    with pytest.raises(ValidationError):
        test_model(arr=arr)
