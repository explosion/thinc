import pytest
import numpy
from numpy.testing import assert_allclose
from hypothesis import given, assume
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers

from ...._classes.window_encode import _get_positions
from ...._classes.window_encode import _get_full_gradients
from ...._classes.window_encode import _get_full_inputs
from ...._classes.window_encode import _compute_hidden_layer
from ...._classes.window_encode import _zero_features_past_sequence_boundaries
from ....ops import NumpyOps

from ...strategies import arrays_BOP_BO
from ...strategies import arrays_OPFI_BI_lengths
from ...strategies import lengths


try:
    import cytoolz as toolz
except ImportError:
    import toolz


@pytest.mark.parametrize(
    'ids_batch', [
        (
            (
                ('the', 'cat', 'sat'),
                ('on', 'the', 'cat')
            ),
        ),
        (
            (
                ('the',),
                ('the',)
            ),
        ),
        (
            (
                ('a', 'b', 'a', 'b', 'd', 'e', 'f'),
            ),
        )
    ]
)
def test_get_positions(ids_batch):
    ids_batch = list(toolz.concat(ids_batch))
    positions = _get_positions(ids_batch)
    for key, idxs in positions.items():
        for i in idxs:
            assert ids_batch[i] == key


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



# I spent ages getting this test right, but the function now works differently...
# We compute into the hidden layer for the types, not the tokens.
# Leaving this here in case I can salvage something from it. If it's still xfail
# in two months time it should be deleted.
# 06/01/2017.
@pytest.mark.xfail
@given(arrays_OPFI_BI_lengths(max_B=10, max_P=10, max_I=10))
def test_compute_hidden_layer(arrays_OPFI_BI_lengths):
    W__OPFI, vectors__BI, lengths = arrays_OPFI_BI_lengths
    # Converting to int saves us from a lot of annoying false positives from
    # floating point arithmetic
    W__OPFI = numpy.asarray(W__OPFI, dtype='int32')
    vectors__BI = numpy.asarray(vectors__BI, dtype='int32')
    O, P, F, I = W__OPFI.shape
    B = sum(lengths)
    assume(F == 5)
    assert sum(lengths) == vectors__BI.shape[0]
    assert vectors__BI.shape[1] == W__OPFI.shape[-1]
    ops = NumpyOps()

    H__BFOP = _compute_hidden_layer(ops, W__OPFI, vectors__BI, lengths)
    assert H__BFOP.shape == (B, F, O, P)

    b = 0
    sequences = ops.unflatten(vectors__BI, lengths)
    for sequence in sequences:
        for i, input__i in enumerate(sequence):
            expected__OPF = numpy.tensordot(W__OPFI, input__i, axes=[[3], [0]])
            assert expected__OPF.shape == (O, P, F)
            expected = expected__OPF.transpose((2, 0, 1))
            computed = H__BFOP[b]
            assert expected.shape == computed.shape
            # Check the centre column first
            assert_allclose(expected[2], computed[2],
                            rtol=1e-4, atol=0.001)
            # Now check the 1 column, which represents the L context.
            if i >= 1:
                assert_allclose(expected[1], computed[1], rtol=1e-4, atol=0.001)
            else:
                assert_allclose(computed[1], 0)
            # Now check the 3 column, which represents the R context.
            if len(sequence)-i >= 2:
                assert_allclose(expected[3], computed[3], rtol=1e-4, atol=0.001)
            else:
                assert_allclose(computed[3], 0)
            # Now check the 4 column, which represents the RR context.
            if len(sequence)-i >= 3:
                assert_allclose(expected[4], computed[4], rtol=1e-4, atol=0.001)
            else:
                assert_allclose(computed[4], 0)
            # Now check the 0 column, which represents the LL context.
            if i >= 2:
                assert_allclose(expected[0], computed[0], rtol=1e-4, atol=0.001)
            else:
                assert_allclose(computed[0], 0)
            b += 1
