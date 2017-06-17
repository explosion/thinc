import numpy
import timeit
from cytoolz import partition_all
from ..ops import NumpyOps


def _get_array_module(array):
    return numpy


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


def RNN(alloc, nO, nI, begin_nonlin):
    Wx, dWx    = alloc((nO, nI), gradient=True)
    Wh, dWh    = alloc((nO, nO), gradient=True)
    b, db      = alloc((nO,),    gradient=True)
    pad, d_pad = alloc((nO,),  gradient=True)
    xp = _get_array_module(Wx)

    def rnn_fwd(Xs):
        Zs = []
        backprops = []
        for X in Xs:
            Y = xp.tensordot(X, Wx, axes=[[1], [1]])
            Y += b
            Z, nonlin_fwd, bp_nonlin = begin_nonlin(Y)
            state = pad
            for t in range(Y.shape[0]):
                Y[t] += Wh.dot(state)
                state = nonlin_fwd(t)
            backprops.append(bp_nonlin)
            Zs.append(Z)

        def rnn_bwd(dZs):
            nonlocal d_pad, dWx, dWh, db
            dXs = []
            for dZ, bp_Z in zip(dZs, backprops):
                dY       = bp_Z(dZ)
                dX       = xp.tensordot(dY,     Wx,     axes=[[1], [0]])
                dX[:-1] += xp.tensordot(dY[1:], Wh,     axes=[[1], [0]])
                d_pad   += dY[0].dot(Wh)
                dWx     += xp.tensordot(dY,     X,      axes=[[0], [0]])
                dWh     += xp.tensordot(dY[1:], Y[:-1], axes=[[0], [0]])
                db      += dY.sum(axis=0)
                dXs.append(dX)
            return dXs
        return Zs, rnn_bwd
    return rnn_fwd


def begin_stepwise_relu(X):
    Y = numpy.zeros(X.shape, dtype='f')
    def relu_fwd(t):
        Y[t] *= X[t]>0
        return Y[t]
    def relu_bwd(dY):
        return dY * (X>0)
    return Y, relu_fwd, relu_bwd


def begin_stepwise_LSTM(gates):
    ops = NumpyOps()
    Hout = numpy.zeros((gates.shape[0], gates.shape[1]), dtype='f')
    cells = numpy.zeros((gates.shape[0], gates.shape[1]), dtype='f')
    d_cells = numpy.zeros(cells.shape, dtype='f')
    d_gates = numpy.zeros(gates.shape, dtype='f')
    def lstm_nonlin_fwd(t):
        ops.lstm(Hout[t], cells[t],
            gates[t], cells[t-1])
        return Hout[t]

    def lstm_nonlin_bwd(dHout):
        for t in range(dHout.shape[0], -1, -1):
            ops.backprop_lstm(d_cells[t], d_gates[t], d_output[t], gates[t],
                cells[t], cells[t-1])
            if t != 0:
                d_cells[t-1] += d_cells[t]
        return d_gates[t]

    return Hout, lstm_nonlin_fwd, lstm_nonlin_bwd


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
    model = RNN(alloc, nO, nI, begin_stepwise_relu)
    start = timeit.default_timer()
    for Xs in batches:
        ys, bp_ys = model(list(Xs))
    end = timeit.default_timer()
    n_samples = n_batch * batch_size
    print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))

