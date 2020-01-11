import numpy
from pydantic import create_model, ValidationError
from thinc.types import Array1d, Array2d, Array3d, Array4d, ArrayNd
import pytest


@pytest.mark.parametrize(
    "arr,arr_type",
    [
        (numpy.zeros(0, dtype=numpy.float32), Array1d),
        (numpy.zeros((0, 0), dtype=numpy.float32), Array2d),
        (numpy.zeros((0, 0, 0), dtype=numpy.float32), Array3d),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.float32), Array4d),
        (numpy.zeros((0), dtype=numpy.float32), ArrayNd),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.float32), ArrayNd),
        (numpy.zeros(0, dtype=numpy.int32), Array1d),
        (numpy.zeros((0, 0), dtype=numpy.int32), Array2d),
        (numpy.zeros((0, 0, 0), dtype=numpy.int32), Array3d),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.int32), Array4d),
        (numpy.zeros(0, dtype=numpy.int32), ArrayNd),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.int32), ArrayNd),
    ],
)
def test_array_validation_valid(arr, arr_type):
    test_model = create_model("TestModel", arr=(arr_type, ...))
    result = test_model(arr=arr)
    assert numpy.array_equal(arr, result.arr)


@pytest.mark.parametrize(
    "arr,arr_type",
    [
        (numpy.zeros((0, 0), dtype=numpy.float32), Array1d),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.float32), Array3d),
    ],
)
def test_array_validation_invalid(arr, arr_type):
    test_model = create_model("TestModel", arr=(arr_type, ...))
    with pytest.raises(ValidationError):
        test_model(arr=arr)
