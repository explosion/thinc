"""Test array type validation using confection's validate_type.

This tests the same array types as test_types.py but uses confection's
validation (which calls __get_pydantic_core_schema__ hooks directly)
instead of pydantic's create_model.
"""

import numpy
import pytest
from confection.validation import validate_type

from thinc.types import (
    Floats1d,
    Floats2d,
    Floats3d,
    Floats4d,
    Ints1d,
    Ints2d,
    Ints3d,
    Ints4d,
)


@pytest.mark.parametrize(
    "arr,arr_type",
    [
        (numpy.zeros(0, dtype=numpy.float32), Floats1d),
        (numpy.zeros((0, 0), dtype=numpy.float32), Floats2d),
        (numpy.zeros((0, 0, 0), dtype=numpy.float32), Floats3d),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.float32), Floats4d),
        (numpy.zeros(0, dtype=numpy.int32), Ints1d),
        (numpy.zeros((0, 0), dtype=numpy.int32), Ints2d),
        (numpy.zeros((0, 0, 0), dtype=numpy.int32), Ints3d),
        (numpy.zeros((0, 0, 0, 0), dtype=numpy.int32), Ints4d),
    ],
)
def test_array_validation_valid(arr, arr_type):
    err = validate_type(arr, arr_type)
    assert err is None, f"Expected valid, got: {err}"


@pytest.mark.parametrize(
    "arr,arr_type",
    [
        (numpy.zeros((0, 0), dtype=numpy.float32), Floats1d),
        (numpy.zeros((0, 0), dtype=numpy.float32), Ints2d),
    ],
)
def test_array_validation_invalid(arr, arr_type):
    err = validate_type(arr, arr_type)
    assert err is not None, f"Expected error for {arr_type.__name__}"


@pytest.mark.parametrize(
    "value,arr_type",
    [
        ("not an array", Floats2d),
        (42, Ints1d),
        (None, Floats3d),
    ],
)
def test_array_validation_wrong_type(value, arr_type):
    err = validate_type(value, arr_type)
    assert err is not None
