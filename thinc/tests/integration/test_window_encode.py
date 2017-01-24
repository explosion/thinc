import pytest
import numpy
from numpy.testing import assert_allclose
from mock import Mock
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers


from ...neural._classes.window_encode import MaxoutWindowEncode
from ...neural._classes.embed import Embed
from ...neural.ops import NumpyOps

from ...neural._classes.window_encode import _get_vector_gradients
from ...neural._classes.window_encode import _get_full_inputs
from ...neural._classes.window_encode import _compute_hidden_layer
from ...neural.ops import NumpyOps

from ..strategies import arrays_BOP_BO
from ..strategies import arrays_OPFI_BI_lengths
from ..strategies import lengths


@pytest.fixture
def nr_out():
    return 5


@pytest.fixture
def ndim():
    return 3


@pytest.fixture
def total_length(positions):
    return sum(len(occurs) for occurs in positions.values())


@pytest.fixture
def ops():
    return NumpyOps()


@pytest.fixture
def nV(positions):
    return max(positions.keys()) + 1


@pytest.fixture
def model(ops, nr_out, ndim, nV):
    model = MaxoutWindowEncode(
                Embed(ndim,ndim, nV), nr_out, ndim, pieces=2, window=2)
    return model


@pytest.mark.slow
@pytest.mark.xfail
def test_forward_succeeds(model, ids, positions, vectors, lengths):
    out, whiches = model._forward(positions, vectors, lengths)


@pytest.mark.slow
@pytest.mark.xfail
def test_predict_batch_succeeds(model, ids, vectors, lengths):
    ids = list(toolz.concat(ids))
    out = model.predict_batch((ids, vectors, lengths))
    assert out.shape == (sum(lengths), model.nr_out)

def test_zero_gradient_makes_zero_finetune(model):
    positions = {10: [0]}
    fwd, finish_update = model.begin_update(positions)
    gradients_BO = model.ops.allocate((1, model.nO))
    finish_update(gradients_BO, sgd=None)
    assert_allclose(model.embed.d_vectors, 0)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.skip
@given(arrays_BOP_BO())
def test_only_one_piece_gets_gradient_if_unique_max(x_BOP_d_BO):
    x_BOP, d_BO = x_BOP_d_BO
    assert len(x_BOP.shape) == 3
    assert x_BOP.shape[0] == d_BO.shape[0]
    assert x_BOP.shape[1] == d_BO.shape[1]
    whiches_BO = numpy.argmax(x_BOP, axis=-1)
    d_BOP = numpy.zeros(x_BOP.shape)

    _get_full_gradients(d_BOP, d_BO, whiches_BO)

    for b in range(x_BOP.shape[0]):
        for o in range(x_BOP.shape[1]):
            num_at_max = sum(x_BOP[b,o] >= x_BOP[b, o, whiches_BO[b,o]])
            if num_at_max == 1.:
                if d_BO[b, o] != 0:
                    assert sum(cell != 0. for cell in d_BOP[b, o]) == 1.
                    assert abs(d_BOP[b, o]).argmax(axis=-1) == x_BOP[b, o].argmax()
                assert sum(d_BOP[b, o]) == d_BO[b, o]


@pytest.mark.slow
@given(arrays_OPFI_BI_lengths(max_B=10, max_P=10, max_I=10))
def test_compute_hidden_layer(arrays_OPFI_BI_lengths):
    W__OPFI, vectors__UI, _ = arrays_OPFI_BI_lengths
    # Converting to int saves us from a lot of annoying false positives from
    # floating point arithmetic
    W__OPFI = numpy.asarray(W__OPFI, dtype='int32')
    vectors__UI = numpy.asarray(vectors__UI, dtype='int32')
    O, P, F, I = W__OPFI.shape
    U = vectors__UI.shape[0]
    # The function works on types, not tokens. But for now assume every token
    # is unique.
    assume(F == 5)
    ops = NumpyOps()

    H__UFOP = _compute_hidden_layer(ops, W__OPFI, vectors__UI)
    assert H__UFOP.shape == (U, F, O, P)

    for u in range(U):
        expected__OPF = numpy.tensordot(W__OPFI, vectors__UI[u], axes=[[3], [0]])
        assert expected__OPF.shape == (O, P, F)
        expected__FOP = expected__OPF.transpose((2, 0, 1))
        computed = H__UFOP[u]
        assert expected__FOP.shape == computed.shape
        assert_allclose(computed, expected__FOP)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.skip
@given(
    lengths().flatmap(lambda lenlen:
        arrays('int32', shape=(lenlen,),
        elements=integers(min_value=1, max_value=10))))
def test_zero_features_past_sequence_boundaries(seq_lengths):
    ops = NumpyOps()
    B = sum(seq_lengths)
    O = 1
    P = 1
    H__BFOP = ops.allocate((B, 5, O, P))
    H__BFOP += 1
    _zero_features_past_sequence_boundaries(H__BFOP, seq_lengths)
    sequences = ops.unflatten(H__BFOP, seq_lengths)
    for sequence in sequences:
        w0 = sequence[0].reshape((5, O, P))
        assert_allclose(w0[0], 0)
        assert_allclose(w0[1], 0)
        if len(sequence) >= 2:
            w1 = sequence[1].reshape((5, O, P))
            assert_allclose(w1[0], 0)
        w_m1 = sequence[-1].reshape((5, O, P))
        assert_allclose(w_m1[3], 0)
        assert_allclose(w_m1[4], 0)
        if len(sequence) >= 2:
            w_m2 = sequence[-2].reshape((5, O, P))
            assert_allclose(w_m2[4], 0)


@pytest.mark.slow
@pytest.mark.xfail
def test_get_full_inputs_zeros_edges():
    B = 11
    F = 5
    I = 3
    output = numpy.zeros((B, F, I))
    vectors = numpy.ones((B, I))
    lengths = [5, 6]
    assert sum(lengths) == B
    _get_full_inputs(output,
        vectors, lengths)
    assert_allclose(output[0, 0], 0)
    assert_allclose(output[0, 1], 0)
    assert_allclose(output[1, 0], 0)
    assert_allclose(output[1, 1], 1)
    assert_allclose(output[3, 0], 1)
    assert_allclose(output[3, 1], 1)
    assert_allclose(output[3, 3], 1)
    assert_allclose(output[3, 4], 0)
    assert_allclose(output[4, 0], 1)
    assert_allclose(output[4, 1], 1)
    assert_allclose(output[4, 3], 0)
    assert_allclose(output[4, 4], 0)
    assert_allclose(output[5, 0], 0)
    assert_allclose(output[5, 1], 0)
    assert_allclose(output[6, 0], 0)
    assert_allclose(output[6, 1], 1)
    assert_allclose(output[6, 3], 1)
    assert_allclose(output[6, 4], 1)
    assert_allclose(output[10, 0], 1)
    assert_allclose(output[10, 1], 1)
    assert_allclose(output[10, 2], 1)
    assert_allclose(output[10, 3], 0)
    assert_allclose(output[10, 4], 0)


