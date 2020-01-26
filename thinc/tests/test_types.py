import numpy
from pydantic import create_model, ValidationError
from thinc.types import Floats1d, Floats2d, Floats3d, Floats4d, FloatsNd
from thinc.types import Ints1d, Ints2d, Ints3d, Ints4d, IntsNd
import pytest


@pytest.mark.parametrize(
    "arr,arr_type",
    [
        (numpy.zeros(0, dtype=numpy.float32), Floats1d),
        (numpy.zeros((0, 0), dtype=numpy.float32), Floats2d),
        (numpy.zeros((0, 0, 0), dtype=numpy.float32), Floats3d),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.float32), Floats4d),
        (numpy.zeros((0), dtype=numpy.float32), FloatsNd),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.float32), FloatsNd),
        (numpy.zeros(0, dtype=numpy.int32), Ints1d),
        (numpy.zeros((0, 0), dtype=numpy.int32), Ints2d),
        (numpy.zeros((0, 0, 0), dtype=numpy.int32), Ints3d),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.int32), Ints4d),
        (numpy.zeros(0, dtype=numpy.int32), IntsNd),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.int32), IntsNd),
    ],
)
def test_array_validation_valid(arr, arr_type):
    test_model = create_model("TestModel", arr=(arr_type, ...))
    result = test_model(arr=arr)
    assert numpy.array_equal(arr, result.arr)


@pytest.mark.parametrize(
    "arr,arr_type",
    [
        (numpy.zeros((0, 0), dtype=numpy.float32), Floats1d),
        (numpy.zeros((0, 0), dtype=numpy.float32), Ints2d),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.float32), IntsNd),
    ],
)
def test_array_validation_invalid(arr, arr_type):
    test_model = create_model("TestModel", arr=(arr_type, ...))
    with pytest.raises(ValidationError):
        test_model(arr=arr)
