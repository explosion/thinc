import numpy
from ..ops import NumpyOps
from ...api import layerize


def LSTM(width, residual=False):
    alloc, params = numpy_params()
    model = _LSTM(alloc, width, width, residual=residual)
    def lstm_fwd(X, drop=0.):
        y, bp_y = model(X)
        
        def lstm_bwd(dy, sgd=None):
            dX = bp_y(dy)
            for param, grad in params:
                sgd(param.ravel(), grad.ravel(), key=id(param))
                grad.fill(0)
            return dX
        return y, lstm_bwd
    return layerize(lstm_fwd)


def BiLSTM(width, residual=False):
    alloc, params = numpy_params()
    model = _BiLSTM(alloc, width, width, residual=residual)
    def lstm_fwd(X, drop=0.):
        y, bp_y = model(X)
        
        def lstm_bwd(dy, sgd=None):
            dX = bp_y(dy)
            for param, grad in params:
                sgd(param.ravel(), grad.ravel(), key=id(param))
                grad.fill(0)
            return dX
        return y, lstm_bwd
    return layerize(lstm_fwd)


def BiRNN(width, residual=False):
    alloc, params = numpy_params()
    model = _BiRNN(alloc, width, width, nonlinearity=begin_stepwise_selu,
                   residual=residual)
    def rnn_fwd(X, drop=0.):
        y, bp_y = model(X)
        
        def rnn_bwd(dy, sgd=None):
            dX = bp_y(dy)
            for param, grad in params:
                sgd(param.ravel(), grad.ravel(), key=id(param))
                grad.fill(0)
            return dX
        return y, rnn_bwd
    return layerize(rnn_fwd)


def RNN(width):
    alloc, params = numpy_params()
    model = _RNN(alloc, width, width, nonlinearity=begin_stepwise_relu, residual=True)
    def rnn_fwd(X, drop=0.):
        y, bp_y = model(X)
        
        def rnn_bwd(dy, sgd=None):
            dX = bp_y(dy)
            for param, grad in params:
                sgd(param.ravel(), grad.ravel(), key=id(param))
                grad.fill(0)
            return dX
        return y, rnn_bwd
    return layerize(rnn_fwd)



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


def _BiRNN(alloc, nO, nI, nG=1, nonlinearity=begin_stepwise_tanh, residual=False):
    l2r_model = _RNN(alloc, nO, nI, nonlinearity, nG=nG, residual=residual)
    r2l_model = _RNN(alloc, nO, nI, nonlinearity, nG=nG, residual=residual)
    xp = numpy
    def birnn_fwd(Xs):
        l2r_Zs, bp_l2r_Zs = l2r_model(Xs) 
        r2l_Zs, bp_r2l_Zs = r2l_model([xp.ascontiguousarray(X[::-1]) for X in Xs]) 
        def birnn_bwd(dZs):
            d_l2r_Zs = []
            d_r2l_Zs = []
            for dZ in dZs:
                l2r = dZ[:, :nO]
                r2l = dZ[:, nO:]
                d_l2r_Zs.append(xp.ascontiguousarray(l2r))
                d_r2l_Zs.append(xp.ascontiguousarray(r2l[::-1]))
            dXs_l2r = bp_l2r_Zs(d_l2r_Zs)
            dXs_r2l = bp_r2l_Zs(d_r2l_Zs)
            dXs = [dXf+dXb[::-1] for dXf, dXb in zip(dXs_l2r, dXs_r2l)]
            return dXs
        Zs = [xp.hstack((Zf, Zb[::-1])) for Zf, Zb in zip(l2r_Zs, r2l_Zs)]
        return Zs, birnn_bwd
    return birnn_fwd


def _BiLSTM(alloc, nO, nI, residual=False):
    return _BiRNN(alloc, nO, nI, nG=4, nonlinearity=begin_stepwise_LSTM,
                 residual=residual)


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


def _RNN(alloc, nO, nI, nonlinearity=begin_stepwise_tanh, nG=1):
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


def _LSTM(alloc, nO, nI):
    return _RNN(alloc, nO, nI, begin_stepwise_LSTM, nG=4)


def begin_stepwise_LSTM(gates, nG):
    ops = NumpyOps()
    xp = ops.xp
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
        if d_output.shape[0] >= 2:
            d_gates[:-1] += xp.tensordot(d_output[1:], Wh,
                            axes=[[1], [1]]).reshape((nN-1, nO, nG))
        for t in range(d_output.shape[0]-1, 0, -1):
            ops.backprop_lstm(d_cells[t], d_cells[t-1], d_gates[t],
                d_output[t], gates[t], cells[t], cells[t-1])
        ops.backprop_lstm(d_cells[0], d_pad, d_gates[0], d_output[0], gates[0],
            cells[0], pad)
        return d_gates.reshape((nN, nO*nG))

    return Hout, lstm_nonlin_fwd, lstm_nonlin_bwd
