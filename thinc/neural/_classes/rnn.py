import numpy
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


def begin_stepwise_tanh(X, nG):
    Y = numpy.zeros(X.shape, dtype='f')
    def tanh_fwd(t):
        Y[t] = numpy.tanh(X[t])
        return Y[t]
    def tanh_bwd(dY):
        return (1-Y**2) * dY
    return Y, tanh_fwd, tanh_bwd


def begin_stepwise_relu(X, nG):
    Y = numpy.zeros(X.shape, dtype='f')
    def relu_fwd(t):
        Y[t] = X[t] * (X[t] > 0)
        return Y[t]
    def relu_bwd(dY):
        return dY * (X>0)
    return Y, relu_fwd, relu_bwd


def RNN(alloc, nO, nI, nonlinearity=begin_stepwise_tanh, nG=1):
    begin_nonlin = nonlinearity
    Wx, dWx    = alloc((nO*nG, nI), gradient=True)
    Wh, dWh    = alloc((nO*nG, nO), gradient=True)
    b, db      = alloc((nO*nG,),    gradient=True)
    pad, d_pad = alloc((nO,),       gradient=True)
    xp = _get_array_module(Wx)
    Wx += xp.random.normal(scale=xp.sqrt(1./nI), size=Wx.size).reshape(Wx.shape)
    Wh += xp.random.normal(scale=xp.sqrt(1./nI), size=Wh.size).reshape(Wh.shape)
    # Initialize forget gates' bias
    if nG == 4:
        b = b.reshape((nO, nG))
        b[:, 0] = 1.
        b = b.reshape((nO * nG,))

    def rnn_fwd(Xs):
        Zs = []
        backprops = []
        for X in Xs:
            Y = xp.tensordot(X, Wx, axes=[[1], [1]])
            Y += b
            Z, nonlin_fwd, bp_nonlin = begin_nonlin(Y, nG)
            state = pad
            for t in range(Y.shape[0]):
                Y[t] += Wh.dot(state)
                state = nonlin_fwd(t)
            backprops.append(bp_nonlin)
            Zs.append(Z)

        def rnn_bwd(dZs):
            nonlocal Zs, d_pad, dWx, dWh, db
            dXs = []
            for X, Z, dZ, bp_Z in zip(Xs, Zs, dZs, backprops):
                dY       = bp_Z(dZ)
                dX       = xp.tensordot(dY,     Wx,     axes=[[1], [0]])
                dX[:-1] += xp.tensordot(dY[1:], Wh,     axes=[[1], [0]])
                d_pad   += dY[0].dot(Wh)
                dWx     += xp.tensordot(dY,     X,      axes=[[0], [0]])
                dWh     += xp.tensordot(dY[1:], Z[:-1], axes=[[0], [0]])
                db      += dY.sum(axis=0)
                dXs.append(dX)
            return dXs
        return Zs, rnn_bwd
    return rnn_fwd


def LSTM(alloc, nO, nI):
    return RNN(alloc, nO, nI, begin_stepwise_LSTM, nG=4)


def begin_stepwise_LSTM(gates, nG):
    ops = NumpyOps()
    gates = gates.reshape((gates.shape[0], gates.shape[1]//nG, nG))
    Hout = numpy.zeros((gates.shape[0], gates.shape[1]), dtype='f')
    cells = numpy.zeros((gates.shape[0], gates.shape[1]), dtype='f')
    pad = numpy.zeros((gates.shape[0],), dtype='f')
    d_pad = numpy.zeros((gates.shape[0],), dtype='f')
    def lstm_nonlin_fwd(t):
        ops.lstm(Hout[t], cells[t],
            gates[t], cells[t-1] if t >= 1 else pad)
        return Hout[t]

    def lstm_nonlin_bwd(d_output):
        d_gates = numpy.zeros(gates.shape, dtype='f')
        d_cells = numpy.zeros(cells.shape, dtype='f')
        for t in range(d_output.shape[0]-1, 0, -1):
            ops.backprop_lstm(d_cells[t], d_cells[t-1], d_gates[t],
                d_output[t], gates[t], cells[t], cells[t-1])
        ops.backprop_lstm(d_cells[0], d_pad, d_gates[0], d_output[0], gates[0],
            cells[0], pad)
        return d_gates.reshape((d_gates.shape[0], d_gates.shape[1] * nG))

    return Hout, lstm_nonlin_fwd, lstm_nonlin_bwd
