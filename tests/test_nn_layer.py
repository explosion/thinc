from numpy.testing import assert_allclose
import numpy as np

from thinc.layer import relu, softmax, d_relu
# Test code inspired by/drawn from Keras, by fchollet
#def test_softmax(weights, nr_wide, nr_out, signal):
#    # Test using a reference implementation of softmax
#    def softmax(values):
#        m = np.max(values)
#        e = np.exp(values - m)
#        return e / np.sum(e)
#
#    x = K.placeholder(ndim=2)
#    exp = s(x)
#    f = K.function([x], [exp])
#    test_values = get_standard_values()
#
#    result = f([test_values])[0]
#    expected = softmax(test_values)
#    assert_allclose(result, expected, rtol=1e-05)
#

def numpy_dot(weights, nr_out, nr_wide, signal):
    W = weights[:-nr_out]
    bias = weights[-nr_out:]
    W = W.reshape((nr_out, nr_wide))
    out = W.dot(signal) + bias
    return out
 

def numpy_relu(weights, nr_out, nr_wide, signal):
    output = numpy_dot(weights, nr_out, nr_wide, signal)
    return output * (output > 0)


def numpy_softmax(weights, nr_out, nr_wide, signal):
    output = numpy_dot(weights, nr_out, nr_wide, signal)
    m = np.max(output)
    e = np.exp(output - m)
    return e / np.sum(e)


def numpy_d_relu(delta, signal_in, weights, nr_out, nr_wide):
    W = weights[:-nr_out].reshape(nr_out, nr_wide)
    return W.T.dot(delta) * (signal_in > 0) 


def test_relu():
    weights = np.asarray([1.0, 2.0, 0.0, 0.0], dtype='float32')
    signal = np.asarray([1.0], dtype='float32').T
    result = relu(weights, 2, 1, signal)
    np_result = numpy_relu(weights, 2, 1, signal)
    assert_allclose(result, np_result, rtol=1e-05)
    assert_allclose(np_result, [1, 2])

    weights = np.asarray([1.0, 2.0, 1.0, 3.0], dtype='float32')
    result = relu(weights, 2, 1, signal)
    assert_allclose(result, [2.0, 5.0], rtol=1e-05)

    weights = np.asarray([1.0, 2.0, -1.0, -5.0], dtype='float32')
    result = relu(weights, 2, 1, signal)
    assert_allclose(result, [0.0, 0.0], rtol=1e-05)


def test_softmax():
    weights = np.asarray([0, 0.1, -0.6, 0.5, 0.9, -0.1, 0.0, 0.0, 0.0], dtype='float32')
    signal = np.asarray([-2.0, 1.0], dtype='float32').T
    result = softmax(weights, 3, 2, signal)
    np_result = numpy_softmax(weights, 3, 2, signal)
    assert_allclose(result, np_result, rtol=1e-05)
    assert_allclose([sum(result)], [1.0])


def test_d_relu():
    nr_out = 3
    nr_wide = 2
    weights = np.asarray([0, 1, 2, 3, 4, 5, 0, 0, 0], dtype='float32')
    delta = np.asarray([0.0, 0.3, 0.6], dtype='float32')
    signal_in = np.asarray([2, 2], dtype='float32')
    
    np_result = numpy_d_relu(delta, signal_in, weights, nr_out, nr_wide)
    result = d_relu(delta, signal_in, weights, nr_out, nr_wide)

    assert_allclose(result, np_result)
