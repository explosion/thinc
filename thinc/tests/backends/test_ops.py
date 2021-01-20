import pytest
import numpy
from hypothesis import given, settings
from hypothesis.strategies import composite, integers
from numpy.testing import assert_allclose
from thinc.api import NumpyOps, CupyOps, Ops, get_ops
from thinc.api import get_current_ops, use_ops
from thinc.api import fix_random_seed
from thinc.api import LSTM
import inspect

from .. import strategies
from ..strategies import ndarrays_of_shape


MAX_EXAMPLES = 10

VANILLA_OPS = Ops(numpy)
NUMPY_OPS = NumpyOps()
BLIS_OPS = NumpyOps(use_blis=True)
CPU_OPS = [NUMPY_OPS, VANILLA_OPS]
XP_OPS = [NUMPY_OPS]
if CupyOps.xp is not None:
    XP_OPS.append(CupyOps())
ALL_OPS = XP_OPS + [VANILLA_OPS]


@pytest.mark.parametrize("op", [NumpyOps, CupyOps])
def test_ops_consistency(op):
    """Test that specific ops don't define any methods that are not on the
    Ops base class and that all ops methods define the exact same arguments."""
    attrs = [m for m in dir(op) if not m.startswith("_")]
    for attr in attrs:
        assert hasattr(Ops, attr)
        method = getattr(op, attr)
        if hasattr(method, "__call__"):
            sig = inspect.signature(method)
            params = [p for p in sig.parameters][1:]
            base_sig = inspect.signature(getattr(Ops, attr))
            base_params = [p for p in base_sig.parameters][1:]
            assert params == base_params, attr
            defaults = [p.default for p in sig.parameters.values()][1:]
            base_defaults = [p.default for p in base_sig.parameters.values()][1:]
            assert defaults == base_defaults, attr
            # If args are type annotated, their types should be the same
            annots = [p.annotation for p in sig.parameters.values()][1:]
            base_annots = [p.annotation for p in base_sig.parameters.values()][1:]
            for i, (p1, p2) in enumerate(zip(annots, base_annots)):
                if p1 != inspect.Parameter.empty and p2 != inspect.Parameter.empty:
                    # Need to check string value to handle TypeVars etc.
                    assert str(p1) == str(p2), attr


@pytest.mark.parametrize("ops", ALL_OPS)
def test_alloc(ops):
    float_methods = (ops.alloc1f, ops.alloc2f, ops.alloc3f, ops.alloc4f)
    for i, method in enumerate(float_methods):
        shape = (1,) * (i + 1)
        arr = method(*shape)
        assert arr.dtype == numpy.float32
        assert arr.ndim == len(shape)
        arr = ops.alloc_f(shape)
        assert arr.dtype == numpy.float32
        assert arr.ndim == len(shape)
    int_methods = (ops.alloc1i, ops.alloc2i, ops.alloc3i, ops.alloc4i)
    for i, method in enumerate(int_methods):
        shape = (1,) * (i + 1)
        arr = method(*shape)
        assert arr.dtype == numpy.int32
        assert arr.ndim == len(shape)
        arr = ops.alloc_i(shape)
        assert arr.dtype == numpy.int32
        assert arr.ndim == len(shape)
    assert ops.alloc(1).ndim == 1


@pytest.mark.parametrize("ops", XP_OPS)
def test_hash_gives_distinct_keys(ops):
    ids = ops.alloc1f(5, dtype="uint64")
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


@pytest.mark.parametrize("ops", CPU_OPS)
def test_seq2col_window_one_small(ops):
    seq = ops.asarray([[1.0], [3.0], [4.0], [5]], dtype="float32")
    cols = ops.seq2col(seq, 1)
    if hasattr(cols, "get"):
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
    baseX = base_ops.alloc(X.shape) + X
    target = base_ops.seq2col(base_ops.asarray(baseX), nW=1)
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
            out[start : start + length].sum(axis=0), X[i] * length, rtol=0.01, atol=0.01
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
    X = ops.softmax(X, inplace=True)
    for row in X:
        assert 0.99999 <= row.sum() <= 1.00001


@pytest.mark.parametrize("cpu_ops", [*CPU_OPS, BLIS_OPS])
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


@pytest.mark.parametrize("ops", XP_OPS)
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


@pytest.mark.parametrize("ops", XP_OPS)
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


def get_lstm_args(depth, dirs, nO, batch_size, nI, draw=None):

    if dirs == 1:
        n_params = (nO * 4) * nI + nO * 4 + nO * 4 * nO + nO * 4
        for _ in range(1, depth):
            n_params += nO * 4 * nO + nO * 4 + nO * 4 * nO + nO * 4
    else:
        n_params = (nO * 2) * nI + nO * 2 + nO * 2 * (nO // 2) + nO * 2
        for _ in range(1, depth):
            n_params += nO * 2 * nO + nO * 2 + nO * 2 * (nO // 2) + nO * 2
        n_params *= 2
    lstm = LSTM(nO, nI, depth=depth, bi=dirs >= 2).initialize()
    assert lstm.get_param("LSTM").size == n_params
    if draw:
        params = draw(ndarrays_of_shape(n_params))
        # For some reason this is crashing hypothesis?
        #size_at_t = draw(ndarrays_of_shape(shape=(batch_size,), lo=1, dtype="int32"))
        size_at_t = numpy.ones(shape=(batch_size,), dtype="int32")
        X = draw(ndarrays_of_shape((int(size_at_t.sum()), nI)))
    else:
        params = numpy.ones((n_params,), dtype="f")
        size_at_t = numpy.ones(shape=(batch_size,), dtype="int32")
        X = numpy.zeros(((int(size_at_t.sum()), nI)))
    H0 = numpy.zeros((depth, dirs, nO // dirs))
    C0 = numpy.zeros((depth, dirs, nO // dirs))
    return (params, H0, C0, X, size_at_t)


@composite
def draw_lstm_args(draw):
    depth = draw(integers(1, 4))
    dirs = draw(integers(1, 2))
    nO = draw(integers(1, 16)) * dirs
    batch_size = draw(integers(1, 6))
    nI = draw(integers(1, 16))
    return get_lstm_args(depth, dirs, nO, batch_size, nI, draw=draw)


@pytest.mark.parametrize("ops", XP_OPS)
@pytest.mark.parametrize(
    "depth,dirs,nO,batch_size,nI",
    [
        (1, 1, 1, 1, 1),
        (1, 1, 2, 1, 1),
        (1, 1, 2, 1, 2),
        (2, 1, 1, 1, 1),
        (2, 1, 2, 2, 2),
        (1, 2, 2, 1, 1),
        (2, 2, 2, 2, 2),
    ],
)
def test_lstm_forward_training(ops, depth, dirs, nO, batch_size, nI):
    reference_ops = Ops()
    params, H0, C0, X, size_at_t = get_lstm_args(depth, dirs, nO, batch_size, nI)
    reference = reference_ops.lstm_forward_training(params, H0, C0, X, size_at_t)
    Y, fwd_state = ops.lstm_forward_training(params, H0, C0, X, size_at_t)
    assert_allclose(fwd_state[2], reference[1][2], atol=1e-4, rtol=1e-3)
    assert_allclose(fwd_state[1], reference[1][1], atol=1e-4, rtol=1e-3)
    assert_allclose(Y, reference[0], atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("ops", XP_OPS)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(args=draw_lstm_args())
def test_lstm_forward_training_fuzz(ops, args):
    params, H0, C0, X, size_at_t = args
    reference_ops = Ops()
    reference = reference_ops.lstm_forward_training(params, H0, C0, X, size_at_t)
    Y, fwd_state = ops.lstm_forward_training(params, H0, C0, X, size_at_t)
    assert_allclose(fwd_state[2], reference[1][2], atol=1e-4, rtol=1e-3)
    assert_allclose(fwd_state[1], reference[1][1], atol=1e-4, rtol=1e-3)
    assert_allclose(Y, reference[0], atol=1e-4, rtol=1e-3)


def test_get_ops():
    assert isinstance(get_ops("numpy"), NumpyOps)
    assert isinstance(get_ops("cupy"), CupyOps)
    with pytest.raises(ValueError):
        get_ops("blah")
    ops = Ops(numpy)
    assert ops.xp == numpy


def test_use_ops():
    class_ops = get_current_ops()
    assert class_ops.name == "numpy"
    with use_ops("numpy"):
        new_ops = get_current_ops()
        assert new_ops.name == "numpy"
    with use_ops("cupy"):
        new_ops = get_current_ops()
        assert new_ops.name == "cupy"
    new_ops = get_current_ops()
    assert new_ops.name == "numpy"


def test_minibatch():
    fix_random_seed(0)
    ops = get_current_ops()
    items = [1, 2, 3, 4, 5, 6]
    batches = ops.minibatch(3, items)
    assert list(batches) == [[1, 2, 3], [4, 5, 6]]
    batches = ops.minibatch((i for i in (3, 2, 1)), items)
    assert list(batches) == [[1, 2, 3], [4, 5], [6]]
    batches = list(ops.minibatch(3, numpy.asarray(items)))
    assert isinstance(batches[0], numpy.ndarray)
    assert numpy.array_equal(batches[0], numpy.asarray([1, 2, 3]))
    assert numpy.array_equal(batches[1], numpy.asarray([4, 5, 6]))
    batches = list(ops.minibatch((i for i in (3, 2, 1)), items, shuffle=True))
    assert batches != [[1, 2, 3], [4, 5], [6]]
    assert len(batches[0]) == 3
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1
    with pytest.raises(ValueError):
        ops.minibatch(10, (i for i in range(100)))
    with pytest.raises(ValueError):
        ops.minibatch(10, True)


def test_multibatch():
    fix_random_seed(0)
    ops = get_current_ops()
    arr1 = numpy.asarray([1, 2, 3, 4])
    arr2 = numpy.asarray([5, 6, 7, 8])
    batches = list(ops.multibatch(2, arr1, arr2))
    assert numpy.concatenate(batches).tolist() == [[1, 2], [5, 6], [3, 4], [7, 8]]
    batches = list(ops.multibatch(2, arr1, arr2, shuffle=True))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    batches = list(ops.multibatch(2, [1, 2, 3, 4], [5, 6, 7, 8]))
    assert batches == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    with pytest.raises(ValueError):
        ops.multibatch(10, (i for i in range(100)), (i for i in range(100)))
    with pytest.raises(ValueError):
        ops.multibatch(10, arr1, (i for i in range(100)), arr2)
