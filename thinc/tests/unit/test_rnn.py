# The RNN module currently has some Python 3-specific code. Comment this
# out until we deal with it.

import numpy
from cytoolz import partition_all
import timeit
import pytest

from ...neural._classes.rnn import _RNN
from ...neural._classes.rnn import _ResidualLSTM
from ...neural._classes.rnn import _BiLSTM
from ...neural._classes.rnn import xp_params
from ...neural._classes.rnn import begin_stepwise_relu
from ...neural._classes.rnn import begin_stepwise_LSTM

def numpy_params():
    return xp_params(numpy)

def test_RNN_allocates_params():
    nO = 1
    nI = 2
    alloc, params = numpy_params()
    model = _RNN(alloc, nO, nI, nonlinearity=begin_stepwise_relu, nG=1)
    for weight, grad in params:
        assert weight.shape == grad.shape
    assert params[0][0].shape == (nO, nI)
    assert params[1][0].shape == (nO, nO)
    assert params[2][0].shape == (nO,)
    assert params[3][0].shape == (nO,)


@pytest.mark.skip
def test_RNN_fwd_bwd_shapes():
    nO = 1
    nI = 2
    alloc, params = numpy_params()
    model = _RNN(alloc, nO, nI, begin_stepwise_relu, nG=1)
    
    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype='f')
    ys, backprop_ys =  model([X])
    dXs = backprop_ys(ys)
    assert numpy.vstack(dXs).shape == numpy.vstack([X]).shape


@pytest.mark.skip
def test_RNN_fwd_correctness():
    nO = 1
    nI = 2
    alloc, params = numpy_params()
    model = _RNN(alloc, nO, nI, begin_stepwise_relu, nG=1)
    (Wx, dWx), (Wh, dWh), (b, db), (pad, d_pad) = params
    
    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype='f')
    Wx[:] = 1.
    b[:] = 0.25
    Wh[:] = 1.
    pad[:] = 0.05

    # Step 1
    # (0.1, 0.1) @ [[1., 1.]] + (1.,) @ [[0.05]]
    # = (0.1 * 1) * 2 + 0.05
    # = (0.25,) + 0.25 bias
    Y = model([X])
    Y = Y[0][0]
    assert list(Y[0]) == [0.5]


@pytest.mark.skip
def test_RNN_learns():
    nO = 2
    nI = 2
    alloc, params = numpy_params()
    model = _RNN(alloc, nO, nI)
    X = numpy.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype='f')
    Y = numpy.asarray([[0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], dtype='f')
    Yhs, bp_Yhs = model([X])
    dXs = bp_Yhs([Yhs[0] - Y])
    loss1 = ((Yhs[0]-Y)**2).sum()
    for param, grad in params:
        param -= 0.001 * grad
        grad.fill(0)
    Yhs, bp_Yhs = model([X])
    dXs = bp_Yhs([Yhs[0] - Y])
    loss2 = ((Yhs[0]-Y)**2).sum()
    assert loss1 > loss2, (loss1, loss2)
    
    

def test_LSTM_learns():
    nO = 2
    nI = 2
    alloc, params = numpy_params()
    model = _ResidualLSTM(alloc, nO)
    X = numpy.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype='f')
    Y = numpy.asarray([[0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], dtype='f')
    Yhs, bp_Yhs = model([X])
    loss1 = ((Yhs[0]-Y)**2).sum()
    Yhs, bp_Yhs = model([X])
    dXs = bp_Yhs([Yhs[0] - Y])
    for param, grad in params:
        param -= 0.001 * grad
        grad.fill(0)
    Yhs, bp_Yhs = model([X])
    dXs = bp_Yhs([Yhs[0] - Y])
    loss2 = ((Yhs[0]-Y)**2).sum()
    assert loss1 > loss2, (loss1, loss2)
    

def test_LSTM_fwd():
    nO = 2
    nI = 2
    alloc, params = numpy_params()
    model = _BiLSTM(alloc, nO, nI)
    
    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype='f')
    ys, backprop_ys =  model([X])
    dXs = backprop_ys(ys)
    assert numpy.vstack(dXs).shape == numpy.vstack([X]).shape
 

@pytest.mark.skip
def test_benchmark_RNN_fwd():
    nO = 128
    nI = 128
    n_batch = 1000
    batch_size = 32
    seq_len = 30
    lengths = numpy.random.normal(scale=1, loc=30, size=n_batch*batch_size)
    lengths = numpy.maximum(lengths, 1)
    batches = []
    uniform_lengths = False
    for batch_lengths in partition_all(batch_size, lengths):
        batch_lengths = list(batch_lengths)
        if uniform_lengths:
            seq_len = max(batch_lengths)
            batch = [numpy.asarray(
                numpy.random.uniform(0., 1., (int(seq_len), nI)), dtype='f')
                for _ in batch_lengths
            ]
        else:
            batch = [numpy.asarray(
                numpy.random.uniform(0., 1., (int(seq_len), nI)), dtype='f')
                for seq_len in batch_lengths
            ]
        batches.append(batch)
    alloc, params = numpy_params()
    model = RNN(alloc, nO, nI, begin_stepwise_relu)
    #model = LSTM(alloc, nO, nI)
    start = timeit.default_timer()
    for Xs in batches:
        ys, bp_ys = model(list(Xs))
        _ = bp_ys(ys)
    end = timeit.default_timer()
    n_samples = n_batch * batch_size
    print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))

