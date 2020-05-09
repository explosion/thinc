import pytest
import numpy
from thinc.api import get_width, Ragged, Padded
from thinc.util import get_array_module, is_numpy_array, to_categorical
from thinc.util import convert_recursive
from thinc.types import ArgsKwargs


@pytest.mark.parametrize(
    "obj,width",
    [
        (numpy.zeros((1, 2, 3, 4)), 4),
        (numpy.array(1), 0),
        (numpy.array([1, 2]), 3),
        ([numpy.zeros((1, 2)), numpy.zeros((1))], 2),
        (Ragged(numpy.zeros((1, 2)), numpy.zeros(1)), 2),
        (
            Padded(
                numpy.zeros((2, 1, 2)),
                numpy.zeros(2),
                numpy.array([1, 0]),
                numpy.array([0, 1]),
            ),
            2,
        ),
        ([], 0),
    ],
)
def test_get_width(obj, width):
    assert get_width(obj) == width


@pytest.mark.parametrize("obj", [1234, "foo", {"a": numpy.array(0)}])
def test_get_width_fail(obj):
    with pytest.raises(ValueError):
        get_width(obj)


def test_array_module_cpu_gpu_helpers():
    xp = get_array_module(0)
    assert hasattr(xp, "ndarray")
    assert is_numpy_array(numpy.zeros((1, 2)))
    assert not is_numpy_array((1, 2))


def test_to_categorical():
    # Test without n_classes
    one_hot = to_categorical(numpy.asarray([1, 2], dtype="i"))
    assert one_hot.shape == (2, 3)
    # From keras
    # https://github.com/keras-team/keras/blob/master/tests/keras/utils/np_utils_test.py
    nc = 5
    shapes = [(1,), (3,), (4, 3), (5, 4, 3), (3, 1), (3, 2, 1)]
    expected_shapes = [
        (1, nc),
        (3, nc),
        (4, 3, nc),
        (5, 4, 3, nc),
        (3, 1, nc),
        (3, 2, 1, nc),
    ]
    labels = [numpy.random.randint(0, nc, shape) for shape in shapes]
    one_hots = [to_categorical(label, nc) for label in labels]
    for label, one_hot, expected_shape in zip(labels, one_hots, expected_shapes):
        assert one_hot.shape == expected_shape
        assert numpy.array_equal(one_hot, one_hot.astype(bool))
        assert numpy.all(one_hot.sum(axis=-1) == 1)
        assert numpy.all(numpy.argmax(one_hot, -1).reshape(label.shape) == label)


def test_convert_recursive():
    is_match = lambda obj: obj == "foo"
    convert_item = lambda obj: obj.upper()
    obj = {
        "a": {("b", "foo"): {"c": "foo", "d": ["foo", {"e": "foo", "f": (1, "foo")}]}}
    }
    result = convert_recursive(is_match, convert_item, obj)
    assert result["a"][("b", "FOO")]["c"] == "FOO"
    assert result["a"][("b", "FOO")]["d"] == ["FOO", {"e": "FOO", "f": (1, "FOO")}]
    obj = {"a": ArgsKwargs(("foo", [{"b": "foo"}]), {"a": ["x", "foo"]})}
    result = convert_recursive(is_match, convert_item, obj)
    assert result["a"].args == ("FOO", [{"b": "FOO"}])
    assert result["a"].kwargs == {"a": ["x", "FOO"]}
