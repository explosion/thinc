import numpy
import timeit
from cytoolz import partition_all
from ..ops import NumpyOps


def numpy_params():
    params = []
    def allocate(shape, gradient=False):
        param = numpy.zeros(shape, dtype='f')
        if not gradient:
            return param
        else:
            d_param = numpy.zeros(shape, dtype='f')
            params.append([param, d_param])
            return param, d_param
    return allocate, params


def _get_array_module(array):
    return numpy


def _flatten(Xs, pad=None):
    assert len(Xs) != 0
    assert isinstance(Xs, list)
    assert len(Xs[0].shape) == 2
    xp = _get_array_module(Xs[0])
    if pad is None:
        padded = Xs
    else:
        if isinstance(pad, int):
            pad = xp.zeros((pad, Xs[0].shape[1]))
        padded = []
        for X in Xs:
            padded.append(X)
            padded.append(pad)
    return xp.vstack(padded), xp.asarray([len(X) for X in Xs], dtype='i')


def _unflatten(X, lengths, pad=None):
    xp = _get_array_module(X)
    Xs = []
    start = 0
    for length in lengths:
        Xs.append(X[start : start + length])
        start += length
        if pad is None:
            pass
        elif isinstance(pad, int):
            start += pad
        else:
            pad += X[start:start+pad.shape[0]]
            start += pad.shape[0]
    return Xs



def RNN(alloc, nO, nI, begin_nonlin):
    Wx, dWx = alloc((nO, nI), gradient=True)
    Wh, dWh = alloc((nO, nO), gradient=True)
    b, db = alloc((nO,), gradient=True)
    pad, d_pad = alloc((1, nI), gradient=True)
    xp = _get_array_module(Wx)
    def rnn_fwd(Xs):
        X, lengths = _flatten(Xs, pad=pad)
        Y = xp.tensordot(X, Wx, axes=[[1], [1]])
        Y += b
        Yf = Y.copy()
        nonlin_fwd, bp_nonlin = begin_nonlin(Y, Yf)
        for t in range(1, Yf.shape[0]):
            Yf[t] += Wh.dot(Yf[t-1])
            nonlin_fwd(t)

        def rnn_bwd(dYf_seqs):
            nonlocal Y, X, Wx, dWx, Wh, dWh, d_pad
            dYf, lengths = _flatten(dYf_seqs, pad=1)
            dY = bp_nonlin(dYf)
            dX = xp.tensordot(dY, Wx, axes=[[1], [0]])
            start = 0
            for length in lengths:
                for t in range(start+length, start, -1):
                    dX[t] += dX[t+1].dot(Wh.T)
                start += length
            dWx += xp.tensordot(dY, X, axes=[[0], [0]])
            dWh += xp.tensordot(dY[1:], X[:-1], axes=[[0], [0]])
            db += dY.sum(axis=0)
            return _unflatten(dX, lengths, pad=d_pad)
        return _unflatten(Yf, lengths, pad=1), rnn_bwd
    return rnn_fwd


def LSTM(alloc, nO, nI):
    Wx, dWx = alloc((nO*4, nI), gradient=True)
    Wh, dWh = alloc((nO*4, nO), gradient=True)
    b, db = alloc((nO*4,), gradient=True)
    pad, d_pad = alloc((1, nI), gradient=True)
    xp = _get_array_module(Wx)
    def lstm_fwd(Xs):
        X, lengths = _flatten(Xs, pad=pad)
        N = X.shape[0]

        gates = xp.tensordot(X, Wx, axes=[[1], [1]])
        gates += b
        gates = gates.reshape((N,  nO, 4))
        
        cells = xp.zeros((N, nO))
        Y = xp.zeros((N, nO))

        for t in range(N):
            ops.lstm(Y[t], cells[t], gates[t], cells[t-1])

        def lstm_bwd(dY_seqs):
            nonlocal Y, X, Wx, dWx, Wh, dWh, d_pad
            dY, lengths = _flatten(dY_seqs, pad=1)
            dX = xp.tensordot(dY, Wx, axes=[[1], [0]])

            d_gates = xp.zeros((N, nO, 4), dtype='f')
            start = 0
            for length in lengths:
                for t in range(start+length, start, -1):
                    d_cells[t] = d_cells[t+1]
                    ops.backprop_lstm(d_cells[t], d_gates[t],
                        dY[t], gates[t], cells[t], cells[t-1])
                start += length
            dX += d_gates.sum(axis=-1)
            d_gates = d_gates.reshape((N, nO*4))
            dWx += xp.tensordot(d_gates, X, axes=[[0], [0]])
            dWh += xp.tensordot(d_gates[1:], X[:-1], axes=[[0], [0]])
            db += dY.sum(axis=0)
            return _unflatten(dX, lengths, pad=d_pad)
        return _unflatten(Yf, lengths, pad=1), lstm_bwd
    return rnn_fwd


def begin_stepwise_relu(X, Y):
    def relu_fwd(t):
        Y[t] *= X[t]>0
    def relu_bwd(dY):
        return dY * (X>0)
    return relu_fwd, relu_bwd


def test_RNN_allocates_params():
    nO = 1
    nI = 2
    alloc, params = numpy_params()
    model = RNN(alloc, nO, nI, begin_stepwise_relu)
    for weight, grad in params:
        assert weight.shape == grad.shape
    assert params[0][0].shape == (nO, nI)
    assert params[1][0].shape == (nO, nO)
    assert params[2][0].shape == (nO,)
    assert params[3][0].shape == (1, nO)


def test_RNN_fwd():
    nO = 1
    nI = 2
    alloc, params = numpy_params()
    model = RNN(alloc, nO, nI, begin_stepwise_relu)
    
    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype='f')
    ys, backprop_ys =  model([X])
    dXs = backprop_ys(ys)
    assert numpy.vstack(dXs).shape == numpy.vstack([X]).shape
 

def test_LSTM_fwd():
    nO = 1
    nI = 2
    alloc, params = numpy_params()
    model = LSTM(alloc, nO, nI)
    
    X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype='f')
    ys, backprop_ys =  model([X])
    dXs = backprop_ys(ys)
    assert numpy.vstack(dXs).shape == numpy.vstack([X]).shape
 

def test_RNN_fwd_speed():
    nO = 128
    nI = 128
    n_batch = 1000
    batch_size = 32
    seq_len = 30
    lengths = numpy.random.normal(scale=10, loc=30, size=n_batch*batch_size)
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
    model = LSTM(alloc, nO, nI)
    start = timeit.default_timer()
    for Xs in batches:
        ys, bp_ys = model(list(Xs))
    end = timeit.default_timer()
    n_samples = n_batch * batch_size
    print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))

