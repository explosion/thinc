import pytest
from hypothesis import given, settings
import numpy
from numpy.testing import assert_almost_equal

from ...neural.ops import NumpyOps
from ...neural._classes.attention import SelfAttention
from ...neural._classes.attention import _ragged_window_dot
from ...neural._classes.attention import softmax_ragged, backprop_softmax_ragged
from ..strategies import arrays_BI_BI_lengths, arrays_BI_lengths

@pytest.fixture
def ops():
    return NumpyOps()

def test_init():
    model = SelfAttention()

@settings(max_examples=5)
@given(
    X_Y_lengths=arrays_BI_BI_lengths(),
)
def test_ragged_window_dot(ops, X_Y_lengths):
    window = 5
    X, Y, lengths = X_Y_lengths
    assert sum(lengths) == X.shape[0] == Y.shape[0]
    assert X.shape[1] == Y.shape[1]
    (output, out_lengths), backprop = _ragged_window_dot(ops, X, Y, lengths, window, window)
    assert sum(out_lengths) > sum(lengths)
    assert sum(out_lengths) < (sum(lengths) * window * 2)
    assert_almost_equal(output[0], ops.xp.dot(X[0], Y[0]), decimal=2)
    assert_almost_equal(output[1], ops.xp.dot(X[0], Y[1]), decimal=2)
    assert_almost_equal(output[2], ops.xp.dot(X[0], Y[2]), decimal=2)
    assert_almost_equal(output[3], ops.xp.dot(X[0], Y[3]), decimal=2)
    assert_almost_equal(output[4], ops.xp.dot(X[0], Y[4]), decimal=2)
    assert_almost_equal(output[5], ops.xp.dot(X[1], Y[0]), decimal=2)
    assert_almost_equal(output[6], ops.xp.dot(X[1], Y[1]), decimal=2)


def test_project_inputs_shapes():
    X = numpy.ones((2, 4), dtype='f')
    lengths = numpy.asarray([1, 1], dtype='i')
    model = SelfAttention(nK=3, nO=5, nI=X.shape[1], nL=1, nR=1)
    (queries, keys, values), backprop = model.project_inputs(X, lengths)
    assert queries.shape == (X.shape[0], model.nK)
    assert keys.shape == (X.shape[0], model.nK)
    assert values.shape == (X.shape[0], model.nO)
    dX = backprop(queries, keys, values)
    assert dX.shape == X.shape


def test_compare_shapes(nK=2, nO=4, nI=5, nL=2, nR=2):
    model = SelfAttention(nK=nK, nO=nO, nI=nI, nL=2, nR=2)
    lengths = numpy.asarray([3, 2, 1], dtype='i')
    N = sum(lengths)
    queries = numpy.ones((N, nK), dtype='f')
    keys = numpy.ones((N, nK), dtype='f')
    attention, backprop = model.compare(queries, keys, lengths)
    d_queries, d_keys = backprop(attention)
    assert d_queries.shape == queries.shape
    assert d_keys.shape == keys.shape


def test_softmax_ragged():
    ops = NumpyOps()
    lengths = numpy.asarray([3, 2, 1], dtype='i')
    X = ops.allocate((10 * sum(lengths),))
    X += ops.xp.random.normal(scale=1, size=X.size)
    Y = softmax_ragged(ops, X, lengths)
    start = 0
    for i, length in enumerate(lengths):
        Y_ = Y[start : start+length]
        assert_almost_equal(Y_.sum(), 1.)
        assert all([0 <= y <= 1.0 for y in Y_])
        start += length

def test_backprop_softmax_ragged():
    ops = NumpyOps()
    lengths = numpy.asarray([3, 2, 1], dtype='i')
    X = ops.allocate((10 * sum(lengths),))
    X += ops.xp.random.normal(scale=1, size=X.size)
    Y = softmax_ragged(ops, X, lengths)
    dY = 1-Y
    dX = backprop_softmax_ragged(ops, dY, Y, lengths)
    assert dX.shape == X.shape


@settings(max_examples=5)
@given(
    X_lengths=arrays_BI_lengths(),
)
def test_rescale(X_lengths):
    X, lengths = X_lengths
    model = SelfAttention(nK=3, nO=5, nI=X.shape[1], nL=1, nR=1)
    (queries, keys, values), get_dX = model.project_inputs(X, lengths)
    attention, backprop_compare = model.compare(queries, keys, lengths)
    output, backprop_rescale = model.rescale(values, attention, lengths,
                                             model.nL, model.nR)

 
