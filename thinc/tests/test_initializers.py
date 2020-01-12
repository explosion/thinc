import pytest
from thinc.api import xavier_uniform_init, zero_init, uniform_init, normal_init
from thinc import registry
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


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("xavier_uniform_init.v0", {"inplace": False}),
        ("zero_init.v0", {"inplace": True}),
        ("uniform_init.v0", {"lo": -0.5, "hi": 0.5, "inplace": False}),
        ("normal_init.v0", {"fan_in": 5, "inplace": True}),
    ],
)
def test_initializer_from_config(name, kwargs):
    """Test that initializers are loaded and configured correctly from registry
    (as partials)."""
    cfg = {"test": {"@initializers": name, **kwargs}}
    func = registry.make_from_config(cfg)["test"]
    func(numpy.ndarray([1, 2, 3, 4], dtype="f"))
