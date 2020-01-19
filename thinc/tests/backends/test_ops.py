import pytest
import numpy
from hypothesis import given, settings
from numpy.testing import assert_allclose
from thinc.api import NumpyOps, CupyOps, Ops, get_ops

from .. import strategies


MAX_EXAMPLES = 10

VANILLA_OPS = Ops(numpy)
NUMPY_OPS = NumpyOps()
CPU_OPS = [NUMPY_OPS, VANILLA_OPS]
XP_OPS = [NUMPY_OPS]
if CupyOps.xp is not None:
    XP_OPS.append(CupyOps())
ALL_OPS = XP_OPS + [VANILLA_OPS]


@pytest.mark.parametrize("ops", XP_OPS)
def test_hash_gives_distinct_keys(ops):
    ids = ops.alloc_f1d(5, dtype="uint64")
    keys = ops.hash(ids, 0)
    assert keys.shape == (5, 4)
    assert keys.dtype == "uint32"
    for i in range(len(ids)):
        for j in range(keys.shape[1]):
            assert keys[i, j] != 0


@pytest.mark.parametrize("ops", XP_OPS)
def test_get_dropout_empty(ops):
    shape = (2, 2)
    drop = 0.0
    mask = ops.get_dropout_mask(shape, drop)
    if drop <= 0.0:
        assert mask[mask == 1.0].all()
    else:
        assert mask[mask != 1.0].all()


@pytest.mark.parametrize("ops", XP_OPS)
def test_get_dropout_not_empty(ops):
    shape = (200, 200)
    drop = 0.5
    mask = ops.get_dropout_mask(shape, drop)
    assert (mask > 1.0).any()
    assert (mask == 0.0).any()
    assert mask.shape == shape


@pytest.mark.parametrize("ops", ALL_OPS)
def test_seq2col_window_one_small(ops):
    seq = ops.asarray([[1.0], [3.0], [4.0], [5]], dtype="float32")
    cols = ops.seq2col(seq, 1)
    if not isinstance(cols, numpy.ndarray):
        cols = cols.get()
    assert_allclose(cols[0], [0.0, 1.0, 3.0])
    assert_allclose(cols[1], [1.0, 3.0, 4.0])
    assert_allclose(cols[2], [3.0, 4.0, 5.0])
    assert_allclose(cols[3], [4.0, 5.0, 0.0])


@pytest.mark.parametrize("ops", XP_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BOP())
def test_maxout(ops, X):
    X = ops.asarray(X)
    expected_best = X.max(axis=-1)
    predicted_best, which = ops.maxout(X)
    ops.xp.testing.assert_allclose(
        expected_best, predicted_best, rtol=0.001, atol=0.001
    )
    # Can't compare 'which' directly, as sort order might be different
    # We could check that using the 'which', we get the right results?


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_seq2col_window_one(ops, X):
    X = ops.asarray(X)
    base_ops = Ops()
    base_ops.xp = ops.xp
    target = base_ops.seq2col(X, nW=1)
    predicted = ops.seq2col(X, nW=1)
    ops.xp.testing.assert_allclose(target, predicted, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("ops", ALL_OPS)
def test_backprop_seq2col_window_one_small(ops):
    cols = ops.asarray(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [2.0, 0.0, 0.0]], dtype="float32"
    )
    expected = [[-1.0], [2.0], [1.0]]
    seq = ops.backprop_seq2col(cols, 1)
    if not isinstance(seq, numpy.ndarray):
        seq = seq.get()
    assert_allclose(seq, expected, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_backprop_seq2col_window_one(ops, X):
    if X.shape[1] % 3:
        return None
    X = ops.asarray(X)
    if ops.xp.abs(X).max() >= 30:
        return None
    base_ops = Ops()
    base_ops.xp = ops.xp
    target = base_ops.backprop_seq2col(X, nW=1)
    predicted = ops.backprop_seq2col(X, nW=1)
    for row in range(target.shape[0]):
        diff = target[row].sum() - predicted[row].sum()
        if diff < -0.1 or diff > 0.1:
            print(row, diff)
            print(target[row])
            print(predicted[row])
    ops.xp.testing.assert_allclose(target, predicted, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("ops", XP_OPS)
def test_seq2col_window_two(ops):
    seq = ops.asarray([[1.0], [2.0], [3.0], [4]], dtype="float32")
    cols = ops.seq2col(seq, 2)
    if not isinstance(cols, numpy.ndarray):
        cols = cols.get()
    assert_allclose(cols[0], [0.0, 0.0, 1.0, 2.0, 3.0])
    assert_allclose(cols[1], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert_allclose(cols[2], [1.0, 2.0, 3.0, 4.0, 0.0])
    assert_allclose(cols[3], [2.0, 3.0, 4.0, 0.0, 0.0])


@pytest.mark.parametrize("ops", XP_OPS)
def test_backprop_seq2col_window_two(ops):
    cols = ops.asarray(
        [
            [0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 0.0],
            [2.0, 3.0, 4.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    # We're summing the values that each row
    # was used as a feature. So row 0 had a
    # gradient of 1 in row 0, 1 in row 2, and
    # 1 in row 3.
    expected = ops.asarray(
        [
            [1 + 1 + 1.0 + 0.0],
            [2.0 + 2.0 + 2.0 + 2.0],
            [3.0 + 3.0 + 3.0 + 3.0],
            [0.0 + 4.0 + 4.0 + 4.0],
        ],
        dtype="f",
    )
    seq = ops.backprop_seq2col(cols, 2)
    ops.xp.testing.assert_allclose(seq, expected, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_backprop_reduce_sum(ops, X):
    X = ops.asarray(X)
    if ops.xp.abs(X).max() >= 5:
        return None
    lengths = ops.asarray([3] * len(X), dtype="i")
    out = ops.backprop_reduce_sum(X, lengths)
    assert out.shape == (sum(lengths), X.shape[1])
    start = 0
    for i, length in enumerate(lengths):
        ops.xp.testing.assert_allclose(
            out[start : start + length].sum(axis=0), X[i] * length, rtol=0.1, atol=0.1
        )
        start += length


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_softmax_sums_to_one(ops, X):
    y = ops.softmax(ops.asarray(X))
    for row in y:
        assert 0.99999 <= row.sum() <= 1.0001


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_softmax_works_inplace(ops, X):
    X = ops.asarray(X)
    ops.softmax(X, inplace=True)
    for row in X:
        assert 0.99999 <= row.sum() <= 1.00001


@pytest.mark.parametrize("cpu_ops", CPU_OPS)
def test_gemm_computes_correctly(cpu_ops):
    W = numpy.zeros((3, 2), dtype="f")
    X = numpy.zeros((4, 2), dtype="f")
    W += numpy.random.uniform(size=W.size).reshape(W.shape)
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    Y = cpu_ops.gemm(X, W, trans2=True)
    expected = numpy.dot(X, W.T)
    assert_allclose(expected, Y, atol=1e-4, rtol=1e-4)
    W = numpy.zeros((2, 3), dtype="f")
    X = numpy.zeros((2, 4), dtype="f")
    W += numpy.random.uniform(size=W.size).reshape(W.shape)
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    Y = cpu_ops.gemm(X, W, trans1=True)
    expected = numpy.dot(X.T, W)
    assert_allclose(expected, Y, atol=1e-4, rtol=1e-4)
    cpu_ops.gemm(X, W, trans1=True, out=Y)


@pytest.mark.parametrize("cpu_ops", CPU_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_clip_low_computes_correctly_for_zero(cpu_ops, X):
    expected = X * (X > 0.0)
    y = cpu_ops.clip_low(X, 0.0)
    assert_allclose(expected, y)
    cpu_ops.clip_low(X, 0.0, inplace=True)


@pytest.mark.parametrize("cpu_ops", CPU_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BOP())
def test_take_which_computes_correctly(cpu_ops, X):
    which = numpy.argmax(X, axis=-1)
    best = cpu_ops.take_which(X, which)
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            assert best[i, j] == max(X[i, j])


@pytest.mark.parametrize("cpu_ops", CPU_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_flatten_unflatten_roundtrip(cpu_ops, X):
    flat = cpu_ops.flatten([x for x in X])
    assert flat.ndim == 1
    unflat = cpu_ops.unflatten(flat, [len(x) for x in X])
    assert_allclose(X, unflat)
    flat2 = cpu_ops.flatten([x for x in X], pad=1, dtype="f")
    assert len(flat2) > len(flat)
    unflat2 = cpu_ops.unflatten(flat2, [len(x) for x in X], pad=1)
    assert_allclose(X, unflat2)


@pytest.mark.parametrize("ops", ALL_OPS)
def test_reduce_sum(ops):
    m = ops.xp.zeros((19, 5), dtype="f")
    m += 1
    lengths = ops.xp.array([5, 5, 3, 6], dtype="i")
    output = ops.reduce_sum(m, lengths)
    assert output.sum() == m.sum(), (output.sum(), m.sum())


@pytest.mark.parametrize(
    "ops", [*XP_OPS, pytest.param(VANILLA_OPS, marks=pytest.mark.xfail("Ops.reduce_max"))]
)
def test_reduce_max_sm(ops):
    X = ops.xp.zeros((6, 3), dtype="f")
    X += ops.xp.random.uniform(-1, 1, X.shape)
    lengths = ops.xp.array([2, 2, 2], dtype="i")
    maxes, which = ops.reduce_max(X, lengths)
    start = 0
    for i, length in enumerate(lengths):
        truth = X[start : start + length].max(axis=0)
        ops.xp.testing.assert_allclose(maxes[i], truth)
        start += length


@pytest.mark.parametrize(
    "ops", [*XP_OPS, pytest.param(VANILLA_OPS, marks=pytest.mark.xfail("Ops.reduce_max"))]
)
def test_reduce_max(ops):
    m = ops.xp.zeros((19, 5), dtype="f")
    m += ops.xp.random.uniform(-1, 1, m.shape)
    lengths = ops.xp.array([5, 5, 3, 6], dtype="i")
    # m[4, 0] = 1
    # m[0, 1] = 2
    # m[1, 3] = 3
    maxes, which = ops.reduce_max(m, lengths)
    start = 0
    for i, length in enumerate(lengths):
        truth = m[start : start + length].max(axis=0)
        ops.xp.testing.assert_allclose(maxes[i], truth)
        start += length


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_softplus(ops, X):
    X = ops.asarray(X)
    Y = ops.softplus(X)
    assert Y.shape == X.shape
    assert not ops.xp.isnan(Y).any()
    assert not (Y == 0).any()
    ops.softplus(X, out=X)


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_backprop_softplus(ops, X):
    X = ops.asarray(X)
    # Test zero gradients result in 0 dX
    zeros = ops.alloc(X.shape)
    dX = ops.backprop_softplus(zeros, X)
    assert dX.shape == X.shape
    assert (dX == 0).all()
    ops.backprop_softplus(zeros, X, out=X)


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_mish(ops, X):
    X = ops.asarray(X)
    Y = ops.mish(X)
    assert Y.shape == X.shape
    assert not ops.xp.isnan(Y).any()


@pytest.mark.parametrize("ops", ALL_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_backprop_mish(ops, X):
    X = ops.asarray(X)
    # Test zero gradients result in 0 dX
    zeros = ops.alloc(X.shape)
    dX = ops.backprop_mish(zeros, X)
    assert dX.shape == X.shape
    assert (dX == 0).all()


def test_get_ops():
    Ops = get_ops("numpy")
    Ops is NumpyOps
    Ops = get_ops("cpu")
    assert Ops is NumpyOps
    Ops = get_ops("cupy")
    assert Ops is CupyOps
    Ops = get_ops("gpu")
    assert Ops is CupyOps
    with pytest.raises(ValueError):
        Ops = get_ops("blah")
    ops = Ops(numpy)
    assert ops.xp == numpy
