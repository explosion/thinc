from __future__ import unicode_literals
import pytest
import numpy
from numpy.testing import assert_allclose
from hypothesis import given, settings

from ...neural._classes.multiheaded_attention import AttentionInputs


@pytest.fixture
def xp():
    return numpy

@pytest.fixture
def lengths():
    return [2, 4, 3]

@pytest.fixture
def nH():
    return 2

@pytest.fixture
def nD():
    return 5

@pytest.fixture
def qkv_data(xp, lengths, nH, nD):
    shape = (sum(lengths), 3, nH, nD)
    data = xp.zeros(shape, dtype="f")
    data += xp.random.uniform(-1., 1., shape)
    return data


@pytest.fixture
def ainputs(qkv_data, lengths):
    return AttentionInputs(qkv_data, lengths)


def test_attention_inputs_init(ainputs):
    assert ainputs.QKV.size == sum(ainputs.lengths) * 3 * ainputs.nH * ainputs.nD


def test_attns_sum_to_one(ainputs):
    attn, backprop_attn = ainputs._get_attn_cpu(1.)
    assert attn.shape == (ainputs.nH, ainputs.nP)
    for h in range(ainputs.nH):
        for s, e, aS, aE in ainputs.slices:
            n = e - s
            seq_attn = attn[h, aS:aE].reshape((n, n))
            assert_allclose(seq_attn.sum(axis=-1), numpy.ones((n,), dtype="f"),
                rtol=1e-5, atol=1e-4)


def test_attn_zero_gradients(ainputs):
    attn, backprop_attn = ainputs._get_attn_cpu(1.)
    d_attn = numpy.zeros(attn.shape, dtype="f")
    dQ, dK = backprop_attn(d_attn)
    assert (dQ == 0).all()
    assert (dK == 0).all()


def test_attn_non_zero_gradients(ainputs):
    attn, backprop_attn = ainputs._get_attn_cpu(1.)
    d_attn = numpy.zeros(attn.shape, dtype="f")
    d_attn += numpy.random.uniform(-1, 1., d_attn.shape)
    dQ1, dK1 = backprop_attn(d_attn)
    attn, backprop_attn = ainputs._get_attn_cpu(1.)
    dQ2, dK2 = backprop_attn(d_attn*2)
    abs_dQ1 = numpy.abs(dQ1).ravel()
    abs_dQ2 = numpy.abs(dQ2).ravel()
    for i in range(abs_dQ1.size):
        assert abs_dQ1[i] < abs_dQ2[i]
    abs_dK1 = numpy.abs(dK1).ravel()
    abs_dK2 = numpy.abs(dK2).ravel()
    for i in range(abs_dK1.size):
        assert abs_dK1[i] < abs_dK2[i]


def get_small_attn_example():
    lengths = [2]
    queries = [
        [1, -1, 1],
        [0, 0, 1]
    ]
    keys = [
        [1, 0, 0],
        [1, -1, 0]
    ]
    values = [
        [1., 1., 1.],
        [1., 1, 1,]
    ]

    dots = [
        [1+0+0, 1+1+0], # q0k0, q0k1
        [0+0+0, 0+0+0]  # q1k0, q1k1
    ]
    # Softmax
    exp1 = 2.718281828459045
    exp2 = 7.38905609893065
    attn = [
        # 2.7 / 10.1 = 0.73, 7.4 / 10.1 = 27
        [exp1 / (exp1+exp2), exp2 / (exp1+exp2)],
        [1. / 2., 1. / 2.]
    ]
    # Let's say we get the gradient:
    d_attn = [[0.0, 1.0], [-0.5, -2.0]]
    y_mul_dy = [[attn[i][j] * d_attn[i][j] for j in range(2)] for i in range(2)]
    d_dots = [[y_mul_dy[i][j] - attn[i][j] * sum(y_mul_dy[i]) for j in range(2)]
              for i in range(2)]
    d_dots = [[d_attn[i][j] * d_dots[i][j] for j in range(2)] for i in range(2)]
    # First get the dots, i.e. [0.73, -1.25]
    #attn_dot_d_attn = [
    #    0.2691089108910891 * 0 + 0.7326732673267328 * 1.,
    #    -0.5*0.5 + -2. * 0.5
    #]
    ## Now to backprop the softmax, 
    #d_dots = [
    #    [attn[0][0] * 0.  - (attn[0][0] * attn_dot_d_attn[0]),
    #     attn[0][1] * 1.  - (attn[0][1] * attn_dot_d_attn[0])],
    #    [attn[1][0] * -.5 - (attn[1][0] * attn_dot_d_attn[1]),
    #     attn[1][1] * -2. - (attn[1][1] * attn_dot_d_attn[1])]
    #]
    #print(d_dots)
    # i.e. [[0. - 0.73 * 0.27], [0.27 - 0.27*0.27],
    #       [-0.25 - (0.5 * 0.73), -1. * (0.5 * 0.73)]]

    d_queries = [
        [sum(d_dots[0][i] * keys[i][j] for i in range(2)) for j in range(3)],
        [sum(d_dots[1][i] * keys[i][j] for i in range(2)) for j in range(3)],
    ]
    d_keys = [
        [sum(d_dots[i][0] * queries[i][j] for i in range(2)) for j in range(3)],
        [sum(d_dots[i][1] * queries[i][j] for i in range(2)) for j in range(3)],
    ]

    data = numpy.array(queries + keys + values, dtype="f")
    QKV = data.reshape((3, 1, 2, 3))
    ainputs = AttentionInputs(QKV, [2], dims=("qkv", "nH", "nN", "nD"))
    d_attn = numpy.array(d_attn, dtype="f").reshape((1, -1))
    expected = {
        "attn": numpy.array(attn, dtype="f").reshape((1, -1)),
        "d_keys": numpy.array(d_keys, dtype="f").reshape((1, 2, 3)),
        "d_queries": numpy.array(d_queries, dtype="f").reshape((1, 2, 3))
    }
    return ainputs, d_attn, expected


def test_attn_forward_backward_small():
    ainputs, d_attn, expected = get_small_attn_example()
    attn, backprop_attn = ainputs.get_attn()
    assert_allclose(attn, expected["attn"])
    d_queries, d_keys = backprop_attn(d_attn)
    assert_allclose(d_queries, expected["d_queries"])
    assert_allclose(d_keys, expected["d_keys"])


def get_small_apply_attn_example():
    lengths = [2]
    queries = [
        [1, -1, 1],
        [0, 0, 1]
    ]
    keys = [
        [1, 0, 0],
        [1, -1, 0]
    ]
    values = [
        [-1., 1., 0.5],
        [1., -1, 2,]
    ]
    # Softmax
    exp1 = 2.718281828459045
    exp2 = 7.38905609893065
    attn = [
        # 2.7 / 10.1 = 0.73, 7.4 / 10.1 = 27
        [exp1 / (exp1+exp2), exp2 / (exp1+exp2)],
        [1. / 2., 1. / 2.]
    ]
    context = [
        [(attn[0][0] * values[0][i] + attn[0][1] * values[1][i])
         for i in range(3)],
        [(attn[1][0] * values[0][i] + attn[1][1] * values[1][i])
         for i in range(3)]
    ]
    d_context = [
        [1., -1., 0.],
        [0.5, 0.5, 0.5]
    ]
    # Calculate d_values
    # context[0,0] came from attn[0,0] * values[0,0] + attn[0,1]*values[1,0]
    # context[1,0] came from attn[1,0] * values[0,0] + attn[1,1]*values[1,0]
    # So d_values[0,0] = attn[0,0] * d_context[0,0] + attn[1,0] * d_context[1,0]
    # So d_values[1,0] = attn[1,0] * d_context[1,0] + attn[1,1] * d_context[1,0]
    d_values = [
        [d_context[0][i] * attn[0][0] + d_context[1][i] * attn[1][0]
         for i in range(3)],
        [d_context[0][i] * attn[0][1] + d_context[1][i] * attn[1][1]
         for i in range(3)]
    ]
    # Calculate d_attn
    # context[0] came from attn[0,0] * values[0] + attn[0,1]*values[1]
    # context[1] came from attn[1,0] * values[0] + attn[1,1]*values[1]
    # So d_attn[0,0] = sum(d_context[0] * values[0]) 
    # d_attn[0,1] = sum(d_context[0] * values[1]) 
    # d_attn[1,0] = sum(d_context[1] * values[0]) 
    # d_attn[1,1] = sum(d_context[1] * values[1]) 
 
    d_attn = [
        [(1*-1 + -1*1 + 0*0.5), (1*1+-1*-1+0*2)],
        [(0.5*-1+0.5*1+0.5*0.5), (0.5*1+0.5*-1+0.5*2)]
    ]

    data = numpy.array(queries + keys + values, dtype="f")
    QKV = data.reshape((3, 1, 2, 3))
    ainputs = AttentionInputs(QKV, [2], dims=("qkv", "nH", "nN", "nD"))
    expected = {
        "attn": numpy.array(attn, dtype="f").reshape((1, -1)),
        "context": numpy.array(context, dtype="f").reshape((1, 2, 3)),
        "d_values": numpy.array(d_values, dtype="f").reshape((1, 2, 3)),
        "d_attn": numpy.array(d_attn, dtype="f")
    }
    d_context = numpy.array(d_context, dtype="f").reshape((1, 2, 3))
    return ainputs, d_context, expected


def test_apply_attn_forward_backward_small():
    ainputs, d_context, expected = get_small_apply_attn_example()
    attn, backprop_attn = ainputs.get_attn()
    assert_allclose(attn, expected["attn"])
    context, backprop_context = ainputs.apply_attn(attn)
    assert_allclose(context, expected["context"])
    d_values, d_attn = backprop_context(d_context)
    d_attn = d_attn.reshape((2, 2))
    assert_allclose(d_values, expected["d_values"], atol=1e-5, rtol=1e-5)
    assert_allclose(d_attn, expected["d_attn"], atol=1e-5, rtol=1e-5)
