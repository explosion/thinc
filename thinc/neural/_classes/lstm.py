"""
This is a batched LSTM forward and backward pass, by Andrej Karpathy
https://gist.github.com/karpathy/587454dc0146a6ae21fc
"""
from __future__ import unicode_literals, print_function # pragma: no cover
import numpy as np # pragma: no cover
import code # pragma: no cover

@describe.on_data(_set_dimensions_if_needed, LSUVinit) # pragma: no cover
@describe.attributes(
    nB=Dimension("Batch size"),
    nI=Dimension("Input size"),
    nO=Dimension("Output size"),
    W=Synapses("Weights matrix",
        lambda obj: (obj.nO, obj.nI),
        lambda W, ops: ops.xavier_uniform_init(W)),
    b=Biases("Bias vector",
        lambda obj: (obj.nO,)),
    d_W=Gradient("W"),
    d_b=Gradient("b")
) # pragma: no cover
class LSTM(Model): # pragma: no cover
    '''
    Code by Andrej Karpathy, here:
    https://gist.github.com/karpathy/587454dc0146a6ae21fc
    '''

    def set_weights(self, data=None, initialize=True):
        # +1 for the biases, which will be the first row of WLSTM
        if initialize:
            self.ops.xavier_init(self.W)
        #self.weights = np.random.randn(input_size + hidden_size + 1,
        #                               4 * hidden_size)
        #self.weights /= np.sqrt(input_size + hidden_size)
        #self.weights[0,:] = 0 # initialize biases to zero
        if self.fancy_forget_bias_init != 0:
            # forget gates get little bit negative bias initially to encourage
            # them to be turned off remember that due to Xavier initialization
            # above, the raw output activations from gates before
            # nonlinearity are zero mean and on order of standard deviation ~1
            self.W[0,hidden_size:2*hidden_size] = self.fancy_forget_bias_init

    def predict(self, sequences):
        X, is_not_end = pack_sequences(sequences)
        do_lstm = begin_lstm_forward(self.W, X.shape[1], X.shape[2])
        out = np.zeros((X.shape[0], X.shape[1], self.nr_out))
        for t in range(X.shape[0]):
            out[t] = do_lstm(X[t], is_not_end[t])
        return unpack_sequences(out, is_not_end)

    def get_fwd_cache(self, X, c0 = None, h0 = None):
        """
        X should be of shape (n,b,input_size), where n = length of sequence,
        b = batch size
        """
        n, b, input_size = X.shape
        d = self.weights.shape[1]/4 # hidden size
        for t in range(n):
            cache = {}
            cache['Hout'] = Hout
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['C'] = C
            cache['Ct'] = Ct
            cache['Hin'] = Hin
            cache['c0'] = c0
            cache['h0'] = h0
        # return C[t], as well so we can continue LSTM with prev state init if needed
        return Hout, C[t], Hout[t], cache

    def begin_update(self, sequences):
        X, is_not_end = pack_sequences(sequences)
        do_lstm = begin_lstm_backward(self.W, X.shape[1], X.shape[2])
        out = np.zeros((X.shape[0], X.shape[1], self.nr_out))
        callbacks = [None for _ in range(X.shape[0])]
        for t in range(X.shape[0]):
            out[t], callbacks[t] = do_lstm(X[t], is_not_end[t])
        callbacks.reverse()
        def finish_update(gradient, optimizer=None, **kwargs):
            for callback in callbacks:
                gradient = callback(gradient)
            return gradient
        return out, finish_update


def begin_LSTM_forward(weights, batch_size, input_size): # pragma: no cover
    d = weights.shape[1]/4 # hidden size
    prevc = np.zeros((batch_size, hidden_size))
    prevh = np.zeros((batch_size, hidden_size))
    # input, forget, output, gate (IFOG)
    IFOG = np.zeros((batch_size, hidden_size * 4))
    C = np.zeros((b, d)) # cell content
    def fwd_lstm_step(X_t, is_not_end):
        # Perform the LSTM forward pass with X as the input
        # hidden representation of the LSTM (gated cell content)
        # concat [x,h] as input to the LSTM
        # (with 1 prepended for bias)
        Hin = np.hstack((np.ones((batch_size, 1)), X_t, prevh))
        # compute all gate activations. dots: (most work is this line)
        np.dot(Hin, self.weights, out=IFOG)
        # non-linearities, computed in-place
        IFO = IFOG[:,:3*d]
        G = IFOG[:,3*d:]
        IFO *= -1
        np.exp(IFO, out=IFO)
        IFO += 1
        IFO **= -1
        np.tanh(G, out=G) # tanh
        I = IFO[:,:d]
        F = IFO[:,d:2*d]
        O = IFO[:,2*d:3*d]
        # compute the cell activation
        C *= F
        C += I * G
        Hout = O * np.tanh(C)
        prevh[:] = Hout
        # Reset states at boundaries.
        prevh *= is_not_end
        IFOG *= is_not_end
        C *= is_not_end
        return Hout
    return fwd_lstm_step


def begin_lstm_backward(weights, d_weights, batch_size, input_size): # pragma: no cover
    C = np.zeros((b,d))
    H = np.zeros((b,d))

    # Perform the LSTM forward pass with X as the input
    xphpb = weights.shape[0] # x plus h plus bias, lol
    Hin = np.zeros((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
    Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
    IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
    IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
    C = np.zeros((n, b, d)) # cell content
    Ct = np.zeros((n, b, d)) # tanh of cell content

    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(self.weights.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros((n,b,input_size))
    dh0 = np.zeros((b, d))
    dc0 = np.zeros((b, d))
    dHout = dHout_in.copy() # make a copy so we don't have any funny side effects
    if dcn is not None:
        dC[n-1] += dcn.copy() # carry over gradients from later
    if dhn is not None:
        dHout[n-1] += dhn.copy()

    def bwd_lstm_step(X):
        # concat [x,h] as input to the LSTM
        prevh = Hout[t-1] if t > 0 else h0
        Hin[t,:,0] = 1 # bias
        Hin[t,:,1:input_size+1] = X[t]
        Hin[t,:,input_size+1:] = prevh
        # compute all gate activations. dots: (most work is this line)
        IFOG[t] = Hin[t].dot(self.weights)
        # non-linearities
        IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
        IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
        # compute the cell activation
        prevc = C[t-1] if t > 0 else c0
        C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
        Ct[t] = np.tanh(C[t])
        Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]

        return Hout[t], finish_update
    return lstm_step


def _get_finish_backward(dC, dprevC, dHout, dprevHout, dIFOGf, dWLSTM, dHin,
        weights, tanhCt, prevc, d): # pragma: no cover
    def finish_update(gradient, optimizer=None, **kwargs):
        dIFOGf[:,2*d:3*d] = tanhCt * dHout
        # backprop tanh non-linearity first then continue backprop
        dC += (1-tanhCt**2) * (IFOGf[:,2*d:3*d] * dHout)
        if t > 0:
            dIFOGf[:,:,d:2*d] = prevc * dC
            dprevc += IFOGf[:,d:2*d] * dC
        else:
            dIFOGf[:,d:2*d] = c0 * dC
            dc0 = IFOGf[:,d:2*d] * dC
        dIFOGf[:,:d] = IFOGf[:,3*d:] * dC
        dIFOGf[:,3*d:] = IFOGf[:,:d] * dC

        # backprop activation functions
        dIFOG[:,3*d:] = (1 - IFOGf[:,3*d:] ** 2) * dIFOGf[:,3*d:]
        y = IFOGf[:,:3*d]
        dIFOG[:,:3*d] = (y*(1.0-y)) * dIFOGf[:,:3*d]

        # backprop matrix multiply
        dWLSTM += np.dot(Hin.transpose(), dIFOG)
        dHin = dIFOG.dot(weights.transpose())

        # backprop the identity transforms into Hin
        dX = dHin[:,1:input_size+1]
        dprevHout += dHin[:,input_size+1:]
        # TODO


# -------------------
# TEST CASES
# -------------------


def checkSequentialMatchesBatch(): # pragma: no cover
    """ check LSTM I/O forward/backward interactions """
    n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
    input_size = 10
    LSTM = LSTMModel(input_size, d) # input size, hidden size
    X = np.random.randn(n,b,input_size)
    h0 = np.random.randn(b,d)
    c0 = np.random.randn(b,d)

    # sequential forward
    cprev = c0
    hprev = h0
    caches = [{} for t in xrange(n)]
    Hcat = np.zeros((n,b,d))
    for t in xrange(n):
        xt = X[t:t+1]
        _, cprev, hprev, cache = LSTM.get_fwd_cache(xt, cprev, hprev)
        caches[t] = cache
        Hcat[t] = hprev

    # sanity check: perform batch forward to check that we get the same thing
    H, _, _, batch_cache = LSTM.get_fwd_cache(X, c0, h0)
    assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

    # eval loss
    wrand = np.random.randn(*Hcat.shape)
    loss = np.sum(Hcat * wrand)
    dH = wrand

    # get the batched version gradients
    BdX, BdWLSTM, Bdc0, Bdh0 = LSTM.backward(dH, batch_cache)

    # now perform sequential backward
    dX = np.zeros_like(X)
    dWLSTM = np.zeros_like(LSTM.weights)
    dc0 = np.zeros_like(c0)
    dh0 = np.zeros_like(h0)
    dcnext = None
    dhnext = None
    for t in reversed(xrange(n)):
        dht = dH[t].reshape(1, b, d)
        dx, dWLSTMt, dcprev, dhprev = LSTM.backward(dht, caches[t], dcnext, dhnext)
        dhnext = dhprev
        dcnext = dcprev

        dWLSTM += dWLSTMt # accumulate LSTM gradient
        dX[t] = dx[0]
        if t == 0:
            dc0 = dcprev
            dh0 = dhprev

    # and make sure the gradients match
    print('Making sure batched version agrees with sequential version: (should all be True)')
    print(np.allclose(BdX, dX))
    print(np.allclose(BdWLSTM, dWLSTM))
    print(np.allclose(Bdc0, dc0))
    print(np.allclose(Bdh0, dh0))


def checkBatchGradient(): # pragma: no cover
    """ check that the batch gradient is correct """
    # lets gradient check this beast
    n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
    input_size = 10
    LSTM = LSTMModel(input_size, d) # input size, hidden size
    X = np.random.randn(n,b,input_size)
    h0 = np.random.randn(b,d)
    c0 = np.random.randn(b,d)

    # batch forward backward
    H, Ct, Ht, cache = LSTM.get_fwd_cache(X, c0, h0)
    wrand = np.random.randn(*H.shape)
    loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
    dH = wrand
    dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, cache)

    def fwd():
        h,_,_,_ = LSTM.forward(X, c0, h0)
        return np.sum(h * wrand)

    # now gradient check all
    delta = 1e-5
    rel_error_thr_warning = 1e-2
    rel_error_thr_error = 1
    tocheck = [X, LSTM.weights, c0, h0]
    grads_analytic = [dX, dWLSTM, dc0, dh0]
    names = ['X', 'WLSTM', 'c0', 'h0']
    for j in xrange(len(tocheck)):
        mat = tocheck[j]
        dmat = grads_analytic[j]
        name = names[j]
        # gradcheck
        for i in xrange(mat.size):
            old_val = mat.flat[i]
            mat.flat[i] = old_val + delta
            loss0 = fwd()
            mat.flat[i] = old_val - delta
            loss1 = fwd()
            mat.flat[i] = old_val

            grad_analytic = dmat.flat[i]
            grad_numerical = (loss0 - loss1) / (2 * delta)

            if grad_numerical == 0 and grad_analytic == 0:
                rel_error = 0 # both are zero, OK.
                status = 'OK'
            elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                rel_error = 0 # not enough precision to check this
                status = 'VAL SMALL WARNING'
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                status = 'OK'
                if rel_error > rel_error_thr_warning: status = 'WARNING'
                if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

            # print stats
            print('%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' % (status, name, repr(np.unravel_index(i, mat.shape)), old_val, grad_analytic, grad_numerical, rel_error))



if __name__ == "__main__": # pragma: no cover
    checkSequentialMatchesBatch()
    checkBatchGradient()
    print('every line should start with OK. Have a nice day!')


#
#
#def forward(W, x, U, h, i, a):
#    # Input and gate computation
#    a[t] = tanh( W[c] * x[t] + U[c] * h[t-1])
#    i[t] = sigma(W[i] * x[t] + U[i] * h[t-1])
#    f[t] = sigma(W[f] * x[t] + U[f] * h[t-1])
#    o[t] = sigma(W[o] * x[t] + U[o] * h[t-1])
#
#    # Memory cell update
#    c[t] = dot(i[t], a[t]) + dot(f[t], c[t-1])
#
#    # Compute output
#    h[t] = dot(o[t], tanh(c[t]))
#
#
#def backward(d_h, tanh_ct, o_t, a_t, i_t, c_t1, f_t):
#    d_o = dot(d_h, tanh_ct)
#    d_c += dot(d_h, o_t, 1-tanh2(c_t))
#    d_i = dot(d_c, a_t)
#    d_a = dot(d_c, i_t)
#    d_f = dot(d_c, c_t1)
#    d_c1 = dot(d_c, f_t)
#
#    d_zt =
