import pytest
import numpy
from thinc.api import NumpyOps, Model, Linear
from thinc.api import with_array, with_padded, with_list, with_ragged, with_getitem
from thinc.types import Padded, Ragged


@pytest.fixture(params=[[], [(10, 2)], [(5, 3), (1, 3)], [(2, 3), (0, 3), (1, 3)]])
def shapes(request):
    return request.param


@pytest.fixture(params=["f", "i"])
def dtype(request):
    return request.param


@pytest.fixture
def ops():
    return NumpyOps()


@pytest.fixture
def list_input(shapes, dtype):
    return [numpy.zeros(shape, dtype="f") for shape in shapes]


@pytest.fixture
def ragged_input(ops, list_input):
    lengths = numpy.array([len(x) for x in list_input], dtype="i")
    if not list_input:
        return Ragged(ops.alloc_f2d(0, 0), lengths)
    else:
        return Ragged(ops.flatten(list_input), lengths)


@pytest.fixture
def padded_input(ops, list_input):
    return ops.list2padded(list_input)


@pytest.fixture
def array_input(ragged_input):
    return ragged_input.data


@pytest.fixture
def padded_data_input(padded_input):
    x = padded_input
    return (x.data, x.size_at_t, x.lengths, x.indices)


@pytest.fixture
def ragged_data_input(ragged_input):
    return (ragged_input.data, ragged_input.lengths)


# As an example operation, lets just trim the last dimension. That
# should catch stuff that confuses the input and output.


def get_array_model():
    def _trim_array_forward(model, X, is_train):
        def backprop(dY):
            return model.ops.alloc_f2d(dY.shape[0], dY.shape[1] + 1)

        return X[:, :-1], backprop

    return with_array(Model("trimarray", _trim_array_forward))


def get_list_model():
    def _trim_list_forward(model, Xs, is_train):
        def backprop(dYs):
            dXs = []
            for dY in dYs:
                dXs.append(model.ops.alloc_f2d(dY.shape[0], dY.shape[1] + 1))
            return dXs

        Ys = [X[:, :-1] for X in Xs]
        return Ys, backprop

    return with_list(Model("trimlist", _trim_list_forward))


def get_padded_model():
    def _trim_padded_forward(model, Xp, is_train):
        def backprop(dYp):
            dY = dYp.data
            dX = model.ops.alloc_f3d(dY.shape[0], dY.shape[1], dY.shape[2] + 1)
            return Padded(dX, dYp.size_at_t, dYp.lengths, dYp.indices)

        assert isinstance(Xp, Padded)
        X = Xp.data
        X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        X = X[:, :-1]
        X = X.reshape((Xp.data.shape[0], Xp.data.shape[1], X.shape[1]))
        return Padded(X, Xp.size_at_t, Xp.lengths, Xp.indices), backprop

    return with_padded(Model("trimpadded", _trim_padded_forward))


def get_ragged_model():
    def _trim_ragged_forward(model, Xr, is_train):
        def backprop(dYr):
            dY = dYr.data
            dX = model.ops.alloc_f2d(dY.shape[0], dY.shape[1] + 1)
            return Ragged(dX, dYr.lengths)

        return Ragged(Xr.data[:, :-1], Xr.lengths), backprop

    return with_ragged(Model("trimragged", _trim_ragged_forward))


def get_checker(inputs):
    if isinstance(inputs, Ragged):
        return assert_raggeds_match
    elif isinstance(inputs, Padded):
        return assert_paddeds_match
    elif isinstance(inputs, list):
        return assert_lists_match
    elif isinstance(inputs, tuple) and len(inputs) == 4:
        return assert_padded_data_match
    elif isinstance(inputs, tuple) and len(inputs) == 2:
        return assert_ragged_data_match
    else:
        return assert_arrays_match


def check_initialize(model, inputs):
    # Just check that these run and don't hit errors. I guess we should add a
    # spy and check that model.layers[0].initialize gets called, but shrug?
    model.initialize()
    model.initialize(X=inputs)
    model.initialize(X=inputs, Y=model.predict(inputs))


def check_transform_produces_correct_output_type_forward(model, inputs, checker):
    # It's pretty redundant to check these three assertions, so if the tests
    # get slow this could be removed. I think it should be fine though?
    outputs = model.predict(inputs)
    assert checker(inputs, outputs)
    outputs, _ = model(inputs, is_train=True)
    assert checker(inputs, outputs)
    outputs, _ = model(inputs, is_train=False)
    assert checker(inputs, outputs)


def check_transform_produces_correct_output_type_backward(model, inputs, checker):
    # It's pretty redundant to check these three assertions, so if the tests
    # get slow this could be removed. I think it should be fine though?
    outputs, backprop = model.begin_update(inputs)
    d_inputs = backprop(outputs)
    assert checker(inputs, d_inputs)


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


def assert_padded_data_match(X, Y):
    return assert_paddeds_match(Padded(*X), Padded(*Y))


def assert_ragged_data_match(X, Y):
    return assert_raggeds_match(Ragged(*X), Ragged(*Y))


def test_with_array_initialize(ragged_input, padded_input, list_input, array_input):
    for inputs in (ragged_input, padded_input, list_input, array_input):
        check_initialize(get_array_model(), inputs)


def test_with_padded_initialize(
    ragged_input, padded_input, list_input, padded_data_input
):
    for inputs in (ragged_input, padded_input, list_input, padded_data_input):
        check_initialize(get_padded_model(), inputs)


def test_with_list_initialize(ragged_input, padded_input, list_input):
    for inputs in (ragged_input, padded_input, list_input):
        check_initialize(get_list_model(), inputs)


def test_with_ragged_initialize(
    ragged_input, padded_input, list_input, ragged_data_input
):
    for inputs in (ragged_input, padded_input, list_input, ragged_data_input):
        check_initialize(get_ragged_model(), inputs)


def test_with_array_forward(ragged_input, padded_input, list_input, array_input):
    for inputs in (ragged_input, padded_input, list_input, array_input):
        checker = get_checker(inputs)
        model = get_array_model()
        check_transform_produces_correct_output_type_forward(model, inputs, checker)


def test_with_list_forward(ragged_input, padded_input, list_input):
    for inputs in (ragged_input, padded_input, list_input):
        checker = get_checker(inputs)
        model = get_list_model()
        check_transform_produces_correct_output_type_forward(model, inputs, checker)


def test_with_padded_forward(ragged_input, padded_input, list_input, padded_data_input):
    for inputs in (ragged_input, padded_input, list_input, padded_data_input):
        checker = get_checker(inputs)
        model = get_padded_model()
        check_transform_produces_correct_output_type_forward(model, inputs, checker)


def test_with_ragged_forward(ragged_input, padded_input, list_input, ragged_data_input):
    for inputs in (ragged_input, padded_input, list_input, ragged_data_input):
        checker = get_checker(inputs)
        model = get_ragged_model()
        check_transform_produces_correct_output_type_forward(model, inputs, checker)


def test_with_array_backward(ragged_input, padded_input, list_input, array_input):
    for inputs in (ragged_input, padded_input, list_input, array_input):
        checker = get_checker(inputs)
        model = get_array_model()
        check_transform_produces_correct_output_type_backward(model, inputs, checker)


def test_with_list_backward(ragged_input, padded_input, list_input):
    for inputs in (ragged_input, padded_input, list_input):
        checker = get_checker(inputs)
        model = get_list_model()
        check_transform_produces_correct_output_type_backward(model, inputs, checker)


def test_with_ragged_backward(
    ragged_input, padded_input, list_input, ragged_data_input
):
    for inputs in (ragged_input, padded_input, list_input, ragged_data_input):
        checker = get_checker(inputs)
        model = get_ragged_model()
        check_transform_produces_correct_output_type_backward(model, inputs, checker)


def test_with_padded_backward(
    ragged_input, padded_input, list_input, padded_data_input
):
    for inputs in (ragged_input, padded_input, list_input, padded_data_input):
        checker = get_checker(inputs)
        model = get_padded_model()
        check_transform_produces_correct_output_type_backward(model, inputs, checker)


def test_with_getitem():
    data = (
        numpy.asarray([[1, 2, 3, 4]], dtype="f"),
        numpy.asarray([[5, 6, 7, 8]], dtype="f"),
    )
    model = with_getitem(1, Linear())
    model.initialize(data, data)
    Y, backprop = model.begin_update(data)
    assert len(Y) == len(data)
    assert numpy.array_equal(Y[0], data[0])  # the other item stayed the same
    assert not numpy.array_equal(Y[1], data[1])
    dX = backprop(Y)
    assert numpy.array_equal(dX[0], data[0])
    assert not numpy.array_equal(dX[1], data[1])
