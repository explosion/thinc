import pytest
from thinc.api import xavier_uniform_init, zero_init, uniform_init, normal_init
import numpy


@pytest.mark.parametrize(
    "init_func", [xavier_uniform_init, zero_init, uniform_init, normal_init]
)
def test_initializer_func_setup(init_func):
    data = numpy.ndarray([1, 2, 3, 4], dtype="f")
    result = init_func(data)
    assert not numpy.array_equal(data, result)
    result = init_func(data, inplace=True)
    assert numpy.array_equal(data, result)
