import pytest
import numpy
from thinc.api import Model, with_array
from thinc.types import Padded, Ragged


@pytest.fixture(params=[[], [(10, 2)], [(5, 3), (1, 3)], [(2, 3), (0, 3), (1, 3)]])
def shapes(request):
    return request.param


@pytest.fixture(params=["f", "i"])
def dtype(request):
    return request.param


@pytest.fixture
def list_input(shapes, dtype):
    return [numpy.zeros(shape, dtype="f") for shape in shapes]


@pytest.fixture
def ragged_input(model, list_input):
    lengths = numpy.array([len(x) for x in list_input], dtype="i")
    if not list_input:
        return Ragged(model.ops.alloc_f2d(0, 0), lengths)
    else:
        return Ragged(model.ops.flatten(list_input), lengths)


@pytest.fixture
def padded_input(model, list_input):
    return model.ops.list2padded(list_input)


@pytest.fixture
def array_input(model, ragged_input):
    return ragged_input.data


# Is there a better way to have a parameterize over a set of fixtures?
@pytest.fixture(params=[0, 1, 2, 3])
def selection(request):
    return request.param


@pytest.fixture
def inputs(list_input, ragged_input, padded_input, array_input, selection):
    return [list_input, ragged_input, padded_input, array_input][selection]


def assert_arrays_match(X, Y):
    assert X.dtype == Y.dtype
    # Transformations are allowed to change last dimension, but not batch size.
    assert X.shape[0] == Y.shape[0]
    return True


def assert_lists_match(X, Y):
    assert isinstance(X, list)
    assert isinstance(Y, list)
    assert len(X) == len(Y)
    for x, y in zip(X, Y):
        assert_arrays_match(x, y)
    return True


def assert_raggeds_match(X, Y):
    assert isinstance(X, Ragged)
    assert isinstance(Y, Ragged)
    assert_arrays_match(X.lengths, Y.lengths)
    assert_arrays_match(X.data, Y.data)
    return True


def assert_paddeds_match(X, Y):
    assert isinstance(X, Padded)
    assert isinstance(Y, Padded)
    assert_arrays_match(X.size_at_t, Y.size_at_t)
    assert X.lengths == Y.lengths
    assert X.indices == Y.indices
    assert X.data.dtype == Y.data.dtype
    assert X.data.shape[1] == Y.data.shape[1]
    assert X.data.shape[0] == Y.data.shape[0]
    return True


@pytest.fixture
def model():
    """
    As an example operation, lets just trim the last dimension. That
    should catch stuff that confuses the input and output.
    """
    return with_array(Model("trimdim", _trim_dim_forward))


def _trim_dim_forward(model, X, is_train):
    def backprop(dY):
        return model.ops.allocate((dY.shape[0], dY.shape[1] + 1))

    return X[:, :-1], backprop


def get_checker(inputs):
    if isinstance(inputs, Ragged):
        return assert_raggeds_match
    elif isinstance(inputs, Padded):
        return assert_paddeds_match
    elif isinstance(inputs, list):
        return assert_lists_match
    else:
        return assert_arrays_match


def test_with_array_produces_correct_output_type_forward(model, inputs):
    checker = get_checker(inputs)
    # It's pretty redundant to check these three assertions, so if the tests
    # get slow this could be removed. I think it should be fine though?
    outputs = model.predict(inputs)
    assert checker(inputs, outputs)
    outputs, _ = model(inputs, is_train=True)
    assert checker(inputs, outputs)
    outputs, _ = model(inputs, is_train=False)
    assert checker(inputs, outputs)
