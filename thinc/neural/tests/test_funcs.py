from __future__ import print_function

from ._funcs_shim import *
import numpy as np
np.random.seed(0)
from numpy.testing import assert_allclose

from hypothesis import given, assume
from hypothesis.strategies import tuples, lists, integers, floats
from hypothesis.extra.numpy import arrays


def lengths(lo=1, hi=10):
    return integers(min_value=lo, max_value=hi)


def shapes(min_rows=1, max_rows=100, min_cols=1, max_cols=100):
    return tuples(lengths(lo=min_rows, hi=max_rows), lengths(lo=min_cols, hi=max_cols))


def ndarrays_of_shape(shape, lo=-1000.0, hi=1000.0):
    return arrays('float32', shape=shape, elements=floats(min_value=lo, max_value=hi))
    

def ndarrays(min_len=0, max_len=10, min_val=-10000000.0, max_val=1000000.0):
    return lengths(lo=1, hi=2).flatmap(
        lambda n: ndarrays_of_shape(n, lo=min_val, hi=max_val))


def matrices(min_rows=1, max_rows=10, min_cols=1, max_cols=10,
             min_value=-10000000.0, max_value=1000000.0):
    return shapes(min_rows=min_rows, max_rows=max_rows,
                  min_cols=min_cols, max_cols=max_cols).flatmap(
                        lambda mn: ndarrays_of_shape(mn, lo=min_value, hi=max_value))


def positive_ndarrays(min_len=0, max_len=10, max_val=100000.0):
    return ndarrays(min_len=min_len, max_len=max_len, min_val=0, max_val=max_val)


def negative_ndarrays(min_len=0, max_len=10, min_val=-100000.0):
    return ndarrays(min_len=min_len, max_len=max_len, min_val=min_val, max_val=-1e-10)

#return lengths.flatmap(
    #    lambda n: arrays('float32', shape=(n,), elements=elements))

def parse_layer(layer_data):
    # Get the first row, excluding the first column
    x = layer_data[0,1:]
    # Get the first column, excluding the first row
    # .ascontiguousarray is support important here!!!!
    b = np.ascontiguousarray(layer_data[1:,0], dtype='float32')
    # Slice out the row and the column used for the X and the bias
    W = layer_data[1:,1:]
    assert x.ndim == 1
    assert b.ndim == 1
    assert b.shape[0] == W.shape[0]
    assert x.shape[0] == W.shape[1]
    assume(not np.isnan(W.sum()))
    assume(not np.isnan(x.sum()))
    assume(not np.isnan(b.sum()))
    assume(not any(np.isinf(val) for val in W.flatten()))
    assume(not any(np.isinf(val) for val in x))
    assume(not any(np.isinf(val) for val in b))
    return x, b, W
 

def split_row(layer_data):
    return (layer_data[0,:], layer_data[:,:])


@given(matrices(min_rows=2, min_cols=2, max_rows=4, max_cols=4,
                min_value=-1.0, max_value=1.0))
def test_dot_plus(layer_data):
    x, b, W = parse_layer(layer_data)
    my_result = np.zeros(shape=(len(b),), dtype='float32')
    call_dot_plus(my_result,
        x, W.flatten(), b, len(b), len(x))
    numpy_result = W.dot(x) + b
    assert_allclose(numpy_result, my_result, atol=1e-5)


@given(matrices(min_rows=2, min_cols=2, max_rows=4, max_cols=4,
                min_value=-1.0, max_value=1.0))
def test_d_dot(layer_data):
    x, top_diff, W = parse_layer(layer_data)
    my_result = np.zeros(shape=(len(x),), dtype='float32')
    call_d_dot(my_result,
        top_diff, W.flatten(), W.shape[0], W.shape[1])
    numpy_result = W.T.dot(top_diff)
    assert_allclose(numpy_result, my_result, rtol=1e-05)


@given(matrices(min_rows=2, min_cols=2, max_rows=4, max_cols=4,
                min_value=-1.0, max_value=1.0))
def test_elu(layer_data):
    x, bias, W = parse_layer(layer_data)
 
    numpy_x = x.copy()
    call_ELU(x, len(x))

    gold = _chainer_elu(numpy_x)
    assert_allclose(x, gold, rtol=1e-05)

def _chainer_elu(x):
    y = x.copy()
    neg_indices = x < 0
    y[neg_indices] = 1.0 * (np.exp(y[neg_indices]) - 1)
    return y


@given(matrices(min_rows=2, min_cols=2, max_rows=4, max_cols=4,
                min_value=-1.0, max_value=1.0))
def test_d_elu(layer_data):
    def _get_gold(gx, x):
        # Code taken from Chainer, but modified to match our API
        neg_indices = x < 0
        gx[neg_indices] *= x[neg_indices] + 1.0
        return gx

    delta, layer_data = split_row(layer_data)
    signal_out, layer_data = split_row(layer_data)
    
    gold = _get_gold(delta.copy(), signal_out.copy())
   
    call_d_ELU(delta, signal_out, len(signal_out))

    assert_allclose(delta, gold, rtol=1e-05)


#def test_normalize():
#    pass
