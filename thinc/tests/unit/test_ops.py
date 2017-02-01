import pytest
import numpy
from hypothesis import given, settings
from numpy.testing import assert_allclose

from .. import strategies
from ...neural.ops import NumpyOps

MAX_EXAMPLES = 10

@pytest.fixture
def ops():
    return NumpyOps()


def test_get_dropout_empty(ops):
    shape = (2,2)
    drop = 0.0
    mask = ops.get_dropout_mask(shape, drop)
    if drop <= 0.0:
        assert mask is None
    else:
        assert mask is not None


def test_get_dropout_not_empty(ops):
    shape = (2,2)
    drop = 0.1
    mask = ops.get_dropout_mask(shape, drop)
    if drop <= 0.0:
        assert mask is None
    else:
        assert mask is not None
    assert mask.shape == shape
    assert all(value >= 0 for value in mask.flatten())


def test_seq2col_window_one(ops):
    seq = ops.asarray([[1.], [3.], [4.], [5]], dtype='float32')
    cols = ops.seq2col(seq, 1)
    assert_allclose(cols[0], [0., 1., 3.])
    assert_allclose(cols[1], [1., 3., 4.])
    assert_allclose(cols[2], [3., 4., 5.])
    assert_allclose(cols[3], [4., 5., 0.])


def test_backprop_seq2col_window_one(ops):
    cols = ops.asarray([
        [0., 0., 0.],
        [-1., 0., 1.],
        [2., 0., 0.],
    ], dtype='float32')
    expected = [[-1.], [2.], [1.]]
    seq = ops.backprop_seq2col(cols, 1)
    assert_allclose(seq, expected)


def test_random_bytes(ops):
    byte_str = ops.random_bytes(10)
    assert len(byte_str) == 10
    assert not all(b == byte_str[0] for b in byte_str)


@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BI())
def test_dropout_forward(ops, X):
    drop_prob = 0.25
    def drop_first_cell(shape, drop_prob_):
        assert drop_prob_ == drop_prob
        drop_mask = numpy.ones(shape)
        drop_mask /= (1. - drop_prob)
        drop_mask[0, 0] = 0.
        return drop_mask

    ops.get_dropout_mask = drop_first_cell
    output, backprop = ops.dropout(X, drop_prob)
    assert output[0, 0] == 0.
    for i in range(1, output.shape[0]):
        for j in range(output.shape[1]):
            assert output[i, j] == X[i, j] * (1. / 0.75)

@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BI())
def test_dropout_backward(ops, X):
    drop_prob = 0.25
    def drop_first_cell(shape, drop_prob_):
        assert drop_prob_ == drop_prob
        drop_mask = numpy.ones(shape)
        drop_mask /= (1. - drop_prob)
        drop_mask[0, 0] = 0.
        return drop_mask

    ops.get_dropout_mask = drop_first_cell
    output, backprop = ops.dropout(X, drop_prob)
    gradient = numpy.ones(output.shape)
    def finish_update(d, *args, **kwargs):
        return d
    output_gradient = backprop(finish_update)(gradient)
    assert output_gradient[0, 0] == 0.
    for i in range(1, output.shape[0]):
        for j in range(output.shape[1]):
            assert output_gradient[i, j] == 1. * (4. / 3.)


@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BI())
def test_softmax_sums_to_one(ops, X):
    y = ops.softmax(X)
    for row in y:
        assert 0.99999 <= row.sum() <= 1.00001


@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BI())
def test_softmax_works_inplace(ops, X):
    ops.softmax(X, inplace=True)
    for row in X:
        assert 0.99999 <= row.sum() <= 1.00001


@settings(max_examples=MAX_EXAMPLES)
@given(W_b_inputs=strategies.arrays_OI_O_BI())
def test_batch_dot_computes_correctly(ops, W_b_inputs):
    W, _, inputs = W_b_inputs
    y = ops.batch_dot(inputs, W)
    expected = numpy.tensordot(inputs, W, axes=[[1], [1]])
    assert_allclose(y, expected)


@settings(max_examples=MAX_EXAMPLES)
@given(arrays_BI_BO=strategies.arrays_BI_BO())
def test_batch_outer_computes_correctly(ops, arrays_BI_BO):
    bi, bo = arrays_BI_BO
    assert bi.shape[0] == bo.shape[0]
    assert len(bi.shape) == len(bo.shape) == 2
    expected = numpy.tensordot(bo, bi, axes=[[0], [0]])
    assert expected.shape == (bo.shape[1], bi.shape[1])
    oi = ops.batch_outer(bo, bi)
    assert_allclose(oi, expected)


@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BI())
def test_norm_computes_correctly(ops, X):
    for row in X:
        assert_allclose([numpy.linalg.norm(row)], [ops.norm(row)],
            rtol=1e-04, atol=0.0001)


@settings(max_examples=MAX_EXAMPLES)
@given(W_b_X=strategies.arrays_OI_O_BI())
def test_dot_computes_correctly(ops, W_b_X):
    W, b, X = W_b_X
    for x in X:
        expected = numpy.dot(W, x)
        y = numpy.dot(W, x)
        assert_allclose(expected, y)


@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BI())
def test_argmax_computes_correctly(ops, X):
    which = ops.argmax(X, axis=-1)
    for i in range(X.shape[0]):
        assert max(X[i]) == X[i, which[i]]


@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BI())
def test_clip_low_computes_correctly_for_zero(ops, X):
    expected = X * (X > 0.)
    y = ops.clip_low(X, 0.)
    assert_allclose(expected, y)


@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BOP())
def test_take_which_computes_correctly(ops, X):
    which = numpy.argmax(X, axis=-1)
    best = ops.take_which(X, which)
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            assert best[i, j] == max(X[i, j])


@settings(max_examples=MAX_EXAMPLES)
@given(X=strategies.arrays_BI())
def test_flatten_unflatten_roundtrip(ops, X):
    flat = ops.flatten([x for x in X])
    assert flat.ndim == 1
    unflat = ops.unflatten(flat, [len(x) for x in X])
    assert_allclose(X, unflat)
