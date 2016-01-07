from __future__ import print_function

from ._funcs_shim import *
import numpy as np
np.random.seed(0)
from numpy.testing import assert_allclose

from hypothesis import given, assume
from hypothesis import strategies as hs


float_list = hs.lists(hs.floats())

@given(float_list, float_list, float_list)
def test_dot_plus(x, W, b):
    assume(W and x and b)
    W = np.asarray(W, dtype='float32')
    x = np.asarray(x, dtype='float32')
    b = np.asarray(b, dtype='float32')
    assume(not np.isnan(W.sum()))
    assume(not np.isnan(x.sum()))
    assume(not np.isnan(b.sum()))
    assume(not any(np.isinf(val) for val in W))
    assume(not any(np.isinf(val) for val in x))
    assume(not any(np.isinf(val) for val in b))
    assume(len(W) > (len(x) * len(b)))
    W = W[:len(x) * len(b)]
    W = W.reshape(len(b), len(x))
    my_result = np.zeros(shape=(len(b),), dtype='float32')
    call_dot_plus(my_result,
        x, W.flatten(), b, len(b), len(x))
    numpy_result = W.dot(x) + b
    assert_allclose(numpy_result, my_result, rtol=1e-05)


@given(float_list, float_list)
def test_d_dot(top_diff, W):
    assume(W and top_diff)
    assume(len(W) > len(top_diff))
    W = np.asarray(W, dtype='float32')
    top_diff = np.asarray(top_diff, dtype='float32')
    assume(not np.isnan(W.sum()))
    assume(not np.isnan(top_diff.sum()))
    assume(not any(np.isinf(val) for val in W))
    assume(not any(np.isinf(val) for val in top_diff))
    
   
    nr_out = len(top_diff)
    nr_wide = int(len(W) / len(top_diff))
    W = W[:nr_out * nr_wide]
    W = W.reshape(nr_out, nr_wide)
    my_result = np.zeros(shape=(nr_wide,), dtype='float32')
    call_d_dot(my_result,
        top_diff, W.flatten(), nr_out, nr_wide)

    numpy_result = W.T.dot(top_diff)
    assert_allclose(numpy_result, my_result, rtol=1e-05)



@given(float_list)
def test_elu(x):
    assume(x)
    assume(not any(np.isnan(val) for val in x))
    assume(not any(np.isinf(val) for val in x))
 
    numpy_x = np.asarray(x, dtype='float32')
    x = np.asarray(x, dtype='float32')
   
    call_ELU(x, len(x))

    gold = _chainer_elu(numpy_x)
    assert_allclose(x, gold, rtol=1e-05)


def _chainer_elu(x):
    y = x.copy()
    neg_indices = x < 0
    y[neg_indices] = 1.0 * (np.exp(y[neg_indices]) - 1)
    return y


@given(float_list, float_list)
def test_d_elu(delta, signal_out):
    def _get_gold(gx, x):
        # Code taken from Chainer, but modified to mat
        neg_indices = x < 0
        gx[neg_indices] *= x[neg_indices] + 1.0
        return gx

    assume(delta)
    assume(signal_out)
    assume(len(delta) == len(signal_out))
    
    delta = np.asarray(delta, dtype='float32')
    signal_out = np.asarray(signal_out, dtype='float32')
 
    assume(not any(np.isnan(val) for val in delta))
    assume(not any(np.isinf(val) for val in signal_out))
 
    gold = _get_gold(delta.copy(), signal_out.copy())
   
    call_d_ELU(delta, signal_out, len(signal_out))

    assert_allclose(delta, gold, rtol=1e-05)


def test_normalize():
    pass
