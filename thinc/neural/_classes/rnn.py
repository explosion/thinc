from ...api import layerize
from ..util import get_array_module
from .model import Model


def begin_stepwise_tanh(X, nG):
    xp = get_array_module(X)
    Y = xp.zeros(X.shape, dtype='f')
    def tanh_fwd(t):
        Y[t] = xp.tanh(X[t])
        return Y[t]
    def tanh_bwd(dY):
        return (1-Y**2) * dY
    return Y, tanh_fwd, tanh_bwd


def begin_stepwise_relu(X, nG):
    xp = get_array_module(X)
    Y = xp.zeros(X.shape, dtype='f')
    def relu_fwd(t):
        Y[t] = X[t] * (X[t] > 0)
        return Y[t]
    def relu_bwd(dY):
        return dY * (X>0)
    return Y, relu_fwd, relu_bwd


def begin_stepwise_LSTM(gates, nG):
    xp = get_array_module(gates)
    nN = gates.shape[0]
    nO = gates.shape[1]//nG
    gates = gates.reshape((nN, nO, nG))
    Hout = xp.zeros((nN, nO), dtype='f')
    cells = xp.zeros((nN, nO), dtype='f')
    pad = xp.zeros((nO,), dtype='f')
    d_pad = xp.zeros((nO,), dtype='f')
    def lstm_nonlin_fwd(t):
        ops.lstm(Hout[t], cells[t],
            gates[t], cells[t-1] if t >= 1 else pad)
        return Hout[t]

    def lstm_nonlin_bwd(d_output, Wh):
        d_gates = xp.zeros(gates.shape, dtype='f')
        d_cells = xp.zeros(cells.shape, dtype='f')
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


def LSTM(width, residual=False, xp=None):
    alloc, params = xp_params(xp)
    model = _ResidualLSTM(alloc, width)
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


def BiLSTM(width, residual=False, xp=None):
    alloc, params = xp_params(xp)
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


def BiRNN(width, residual=False, xp=None):
    alloc, params = xp_params(xp)
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


def RNN(width, residual=True, xp=None):
    alloc, params = xp_params(xp)
    model = _RNN(alloc, width, width, nonlinearity=begin_stepwise_relu,
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


def xp_params(xp=None):
    if xp is None:
        xp = Model.Ops.xp
    params = []
    def allocate(shape, gradient=False):
        param = xp.zeros(shape, dtype='f')
        if not gradient:
            return param
        else:
            d_param = xp.zeros(shape, dtype='f')
            params.append([param, d_param])
            return param, d_param
    return allocate, params


def _BiRNN(alloc, nO, nI, nG=1, nonlinearity=begin_stepwise_tanh, residual=False):
    #l2r_model = _RNN(alloc, nO, nI, nonlinearity, nG=nG, residual=residual)
    #r2l_model = _RNN(alloc, nO, nI, nonlinearity, nG=nG, residual=residual)
    assert nO == nI
    l2r_model = _ResidualLSTM(alloc, nI)
    r2l_model = _ResidualLSTM(alloc, nI)
    def birnn_fwd(Xs):
        xp = get_array_module(Xs[0])
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

def _RNN(alloc, nO, nI, nonlinearity=begin_stepwise_tanh, nG=1,
        residual=False):
    begin_nonlin = nonlinearity
    if not residual:
        Wx, dWx    = alloc((nO*nG, nI), gradient=True)
    Wh, dWh    = alloc((nO*nG, nO), gradient=True)
    b, db      = alloc((nO*nG,),    gradient=True)
    pad, d_pad = alloc((nO,),       gradient=True)
    xp = get_array_module(Wh)
    if not residual:
        Wx += xp.random.normal(scale=xp.sqrt(1./nI), size=Wx.size).reshape(Wx.shape)
    Wh += xp.random.normal(scale=xp.sqrt(1./nI), size=Wh.size).reshape(Wh.shape)
    # Initialize forget gates' bias
    if nG == 4:
        b = b.reshape((nO, nG))
        b[:, 0] = 3.
        b = b.reshape((nO * nG,))

    nonlocals = [[], d_pad, dWx, dWh, db]

    def rnn_fwd(Xs):
        nonlocals[0] = []
        Zs = nonlocals[0]
        Ys = []
        backprops = []
        for X in Xs:
            if residual:
                Y = xp.zeros((X.shape[0], nO*nG), dtype='f')
            else:
                Y = xp.tensordot(X, Wx, axes=[[1], [1]])
            Y += b
            Z, nonlin_fwd, bp_nonlin = begin_nonlin(Y, nG)
            state = pad
            for t in range(Y.shape[0]):
                Y[t] += Wh.dot(state)
                state = nonlin_fwd(t)
                if residual:
                    Z[t] += X[t]
                    state += X[t]
            backprops.append(bp_nonlin)
            Ys.append(Y)
            Zs.append(Z)

        def rnn_bwd(dZs):
            Zs, d_pad, dWx, dWh, db = nonlocals
            dXs = []
            for X, Z, dZ, bp_Z in zip(Xs, Zs, dZs, backprops):
                dY = bp_Z(dZ, Wh)
                if residual:
                    dX = dZ.copy()
                else:
                    dX   = xp.tensordot(dY,     Wx,     axes=[[1], [0]])
                    dWx += xp.tensordot(dY,     X,      axes=[[0], [0]])
                if dY.shape[0] >= 2:
                    dWh += xp.tensordot(dY[1:], Z[:-1], axes=[[0], [0]])
                db  += dY.sum(axis=0)
                dXs.append(dX)
            return dXs
        return Zs, rnn_bwd
    return rnn_fwd


def _ResidualLSTM(alloc, nI):
    nO = nI
    nG = 4
    W, dW      = alloc((nO*nG, nO), gradient=True)
    b, db      = alloc((nO*nG,),    gradient=True)
    pad = alloc((nO,))
    xp = get_array_module(W)
    W += xp.random.normal(scale=xp.sqrt(1./nI), size=W.size).reshape(W.shape)
    # Initialize forget gates' bias
    b = b.reshape((nO, nG))
    b[:, 0] = 3.
    b = b.reshape((nO * nG,))

    nonlocals = [dW, db]
    ops = Model.ops

    def lstm_fwd(Xs):
        batch_gates = []
        batch_cells = []
        batch_Houts = []
        for X in Xs:
            nN = X.shape[0]
            gates = xp.zeros((nN, nO * nG), dtype='f')
            Hout = xp.zeros((nN, nO), dtype='f')
            cells = xp.zeros((nN, nO), dtype='f')
 
            gates += b
            for t in range(nN):
                gates[t] += W.dot(Hout[t-1] if t >= 1 else pad)
                ops.lstm(Hout[t], cells[t], gates[t],
                    cells[t-1] if t >= 1 else pad)
                Hout[t] += X[t]
            batch_gates.append(gates)
            batch_cells.append(cells)
            batch_Houts.append(Hout)

        def lstm_bwd(d_Houts):
            dW, db = nonlocals
            dXs = []
            for X, gates, cells, dH in zip(Xs, batch_gates, batch_cells, d_Houts):
                nN = X.shape[0]
                d_gates = xp.zeros((nN, nO * nG), dtype='f')
                d_cells = xp.zeros((nN, nO), dtype='f')
                for t in range(nN-1, 0, -1):
                    ops.backprop_lstm(d_cells[t], d_cells[t-1], d_gates[t],
                        dH[t], gates[t], cells[t], cells[t-1])
                    dH[t-1] += xp.tensordot(d_gates[t], W, axes=[[0], [0]])
                if nN >= 2:
                    dW += xp.tensordot(d_gates[1:], dH[:-1], axes=[[0], [0]])
                db  += d_gates.sum(axis=0)
                dXs.append(dH.copy())
            return dXs
        return batch_Houts, lstm_bwd
    return lstm_fwd


def lstm_fwd(Xs_lengths, W, b):
    Xs, lengths = Xs_lengths
    timesteps = []
    Hp = pad
    Cp = pad
    for t in max(lengths):
        Xt = _make_timestep(Xs, lengths, t)
        Gt = xp.zeros((nB, nO, nG), dtype='f')
        Ht = xp.zeros((nB, nO), dtype='f')
        Ct = xp.zeros((nN, nO), dtype='f')

        Gt += b
        Gt += W.dot(Hp)
        ops.lstm(Ht, Ct, Gt,
            Cp)
        Ht += Xt
        timesteps.append((Xt, Gt, Ct))
        _write_timestep(Hs, lengths, t, Ht)
        Cp = Ct
        Hp = Ht

    def lstm_bwd(dHs):
        dXs = []
        Cp = pad
        Hp = pad
        dHp = xp.zeros((nB, nO), dtype='f')
        dXs = xp.zeros(Xs.shape, dtype='f')
        for t, (Xt, Gt, Ct) in reversed(enumerate(timesteps)):
            dHt = dHp + _make_timestep(Hs, lengths, t)
            dGt.fill(0)
            dCt.fill(0)
            ops.backprop_lstm(dCt, dCp, dGt,
                dHt, Gt, Ct, Cp)
            dHp = dG.dot(W.T)
            dW += xp.tensordot(dGn, dHt, axes=[[0], [0]])
            db += dGt.sum(axis=0)
            _write_timestep(dXs, lengths, t, dHt)
            dCt, dCp = dCp, dCt
        return dXs
    return Hs, lstm_bwd


def _make_timestep(Xs, lengths, t):
    xp = get_array_module(Xs)
    n = 0
    for i, length in enumerate(lengths):
        n += length < t
    output = xp.zeros((n,) + Xs.shape[1:], dtype=Xs.dtype)
    start = 0
    i = 0
    for length in lengths:
        if t < length:
            output[i] = Xs[start + t]
            i += 1
        start += length
    return output


def _write_timestep(Xs, lengths, t, timestep):
    xp = get_array_module(Xs)
    start = 0
    i = 0
    for length in lengths:
        if t < length:
            Xs[start + t] = timestep[i]
            i += 1
        start += length
 

#
#def begin_stepwise_LSTM(gates, nG):
#    ops = NumpyOps()
#    xp = ops.xp
#    nN = gates.shape[0]
#    nO = gates.shape[1]//nG
#    gates = gates.reshape((nN, nO, nG))
#    Hout = numpy.zeros((nN, nO), dtype='f')
#    cells = numpy.zeros((nN, nO), dtype='f')
#    pad = numpy.zeros((nO,), dtype='f')
#    d_pad = numpy.zeros((nO,), dtype='f')
#    def lstm_nonlin_fwd(t):
#        ops.lstm(Hout[t], cells[t],
#            gates[t], cells[t-1] if t >= 1 else pad)
#        return Hout[t]
#
#    def lstm_nonlin_bwd(d_output, Wh):
#        d_gates = numpy.zeros(gates.shape, dtype='f')
#        d_cells = numpy.zeros(cells.shape, dtype='f')
#        if d_output.shape[0] >= 2:
#            d_gates[:-1] += xp.tensordot(d_output[1:], Wh,
#                            axes=[[1], [1]]).reshape((nN-1, nO, nG))
#        for t in range(d_output.shape[0]-1, 0, -1):
#            ops.backprop_lstm(d_cells[t], d_cells[t-1], d_gates[t],
#                d_output[t], gates[t], cells[t], cells[t-1])
#        ops.backprop_lstm(d_cells[0], d_pad, d_gates[0], d_output[0], gates[0],
#            cells[0], pad)
#        return d_gates.reshape((nN, nO*nG))
#
#    return Hout, lstm_nonlin_fwd, lstm_nonlin_bwd
#
#
