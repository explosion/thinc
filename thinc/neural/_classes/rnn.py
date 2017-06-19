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
    def tanh_bwd(dY, Wh):
        dX = dY.copy()
        for t in range(dY.shape[0]-1, 0, -1):
            dX[t] *= 1-Y[t]**2
            dX[t-1] += dX[t].dot(Wh.T)
        dX[0] *= 1-Y[0]**2
        return dX
    return Y, tanh_fwd, tanh_bwd


def begin_stepwise_relu(X, nG):
    Y = numpy.zeros(X.shape, dtype='f')
    def relu_fwd(t):
        Y[t] = X[t] * (X[t] > 0)
        return Y[t]
    def relu_bwd(dY, Wh):
        dX = dY.copy()
        for t in range(dY.shape[0]-1, 0, -1):
            dX[t] *= X[t]>0
            dX[t-1] += dX[t].dot(Wh.T)
        dX[0] *= X[0]>0
        return dX
    return Y, relu_fwd, relu_bwd


def begin_stepwise_selu(X, nG):
    Y = numpy.zeros(X.shape, dtype='f')
    ops = NumpyOps()
    def selu_fwd(t):
        Y[t] = X[t]
        ops.selu(Y[t], inplace=True)
        return Y[t]
    def selu_bwd(dY, Wh):
        dX = dY.copy()
        for t in range(dY.shape[0]-1, 0, -1):
            ops.backprop_selu(dX[t], X[t], inplace=True)
            dX[t-1] += dX[t].dot(Wh.T)
        ops.backprop_selu(dX[0], X[0], inplace=True)
        return dX
    return Y, selu_fwd, selu_bwd


def RNN(alloc, nO, nI, nonlinearity=begin_stepwise_tanh, nG=1,
        residual=False):
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
        b[:, 0] = 3.
        b = b.reshape((nO * nG,))

    def rnn_fwd(Xs):
        Zs = []
        Ys = []
        backprops = []
        for X in Xs:
            Y = xp.tensordot(X, Wx, axes=[[1], [1]])
            Y += b
            Z, nonlin_fwd, bp_nonlin = begin_nonlin(Y, nG)
            state = pad
            for t in range(Y.shape[0]):
                Y[t] += Wh.dot(state)
                state = nonlin_fwd(t)
            if residual:
                Z += X
            backprops.append(bp_nonlin)
            Ys.append(Y)
            Zs.append(Z)

        def rnn_bwd(dZs):
            nonlocal Zs, d_pad, dWx, dWh, db
            dXs = []
            for X, Z, dZ, bp_Z in zip(Xs, Zs, dZs, backprops):
                dY = bp_nonlin(dZ, Wh)
                dX   = xp.tensordot(dY,     Wx,     axes=[[1], [0]])
                dWx += xp.tensordot(dY,     X,      axes=[[0], [0]])
                dWh += xp.tensordot(dY[1:], Z[:-1], axes=[[0], [0]])
                db  += dY.sum(axis=0)
                if residual:
                    dX += dZ
                dXs.append(dX)
            return dXs
        return Zs, rnn_bwd
    return rnn_fwd


def LSTM(alloc, nO, nI, residual=False):
    return RNN(alloc, nO, nI, begin_stepwise_LSTM, nG=4,
               residual=residual)


def BiLSTM(alloc, nO, nI, nG=4, residual=False):
    l2r_model = RNN(alloc, nO//2, nI, begin_stepwise_LSTM, nG=4)
    r2l_model = RNN(alloc, nO//2, nI, begin_stepwise_LSTM, nG=4)
    xp = numpy
    def bilstm_fwd(Xs):
        l2r_Zs, bp_l2r_Zs = l2r_model(Xs) 
        r2l_Zs, bp_r2l_Zs = r2l_model([X[::-1] for X in Xs]) 
        def bilstm_bwd(dZs):
            d_l2r_Zs = []
            d_r2l_Zs = []
            for dZ in dZs:
                l2r = dZ[:, :nO//2]
                r2l = dZ[:, nO//2:]
                r2l = r2l[::-1]
                l2r_Zs.append(xp.ascontiguousarray(l2r))
                r2l_Zs.append(xp.ascontiguousarray(r2l))
            dXs_l2r = bp_l2r_Zs(l2r_Zs)
            dXs_r2l = bp_r2l_Zs(r2l_Zs)
            dXs = [dXf+dXb[::-1] for dXf, dXb in zip(dXs_l2r, dXs_r2l)]
            return dXs
        Zs = [xp.hstack((Zf, Zb[::-1])) for Zf, Zb in zip(l2r_Zs, r2l_Zs)]
        return Zs, bilstm_bwd
    return bilstm_fwd


def begin_stepwise_LSTM(gates, nG):
    ops = NumpyOps()
    nN = gates.shape[0]
    nO = gates.shape[1]//nG
    gates = gates.reshape((nN, nO, nG))
    Hout = numpy.zeros((nN, nO), dtype='f')
    cells = numpy.zeros((nN, nO), dtype='f')
    pad = numpy.zeros((nO,), dtype='f')
    d_pad = numpy.zeros((nO,), dtype='f')
    def lstm_nonlin_fwd(t):
        ops.lstm(Hout[t], cells[t],
            gates[t], cells[t-1] if t >= 1 else pad)
        return Hout[t]

    def lstm_nonlin_bwd(d_output, Wh):
        d_gates = numpy.zeros(gates.shape, dtype='f')
        d_cells = numpy.zeros(cells.shape, dtype='f')
        for t in range(d_output.shape[0]-1, 0, -1):
            ops.backprop_lstm(d_cells[t], d_cells[t-1], d_gates[t],
                d_output[t], gates[t], cells[t], cells[t-1])
            d_gates[t-1] += d_output[t].dot(Wh.T).reshape((nO, nG))
        ops.backprop_lstm(d_cells[0], d_pad, d_gates[0], d_output[0], gates[0],
            cells[0], pad)
        return d_gates.reshape((nN, nO*nG))

    return Hout, lstm_nonlin_fwd, lstm_nonlin_bwd
