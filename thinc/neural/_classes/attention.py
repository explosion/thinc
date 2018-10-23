import math
from ... import describe
from ...describe import Dimension, Synapses, Gradient
from .model import Model


class SelfAttention(Model):
    def __init__(self, nK=None, nO=None, nL=5, nR=5, **kwargs):
        self.nK = nK
        self.nO = nO
        self.nL = nL
        self.nR = nR
        Model.__init__(self, **kwargs)

    def begin_update(self, X_lengths):
        X, lengths = X_lengths
        
        (queries, keys, values), get_dX = self.project_inputs(X)
        (attention, alengths), backprop_compare = self.compare(queries, keys, lengths)
        output, backprop_rescale = self.rescale(values, attention, alengths)

        def self_attention_finish_update(d_output, sgd=None):
            d_values, d_attention = backprop_rescale(d_output)
            d_queries, d_keys = backprop_compare(d_attention)
            dX = get_dX(d_queries, d_keys, d_values)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradients, key=self._mem.key)
            return dX
        return output, self_attention_finish_update

    def project_inputs(self, X, lengths):
        # Let's say X is (25, 300)
        # If nK=32 and nO=64, we need to project down to 32+32+64=128
        # So we have a weight matrix of shape (300, 128)
        Y = self.ops.gemm(X, self.W) # (25, 128)
        queries = Y[:, :self.nK] # Shape (25, 32)
        keys = Y[:, self.nK:self.nK*2]
        # This will be shape (sum(kv_lengths), 64)
        values = Y[:, self.nK*2:]
        def backprop_get_inputs(d_queries, d_keys, d_values):
            dY = self.ops.xp.hstack((d_queries, d_keys, d_values)) # (25, 128)
            # ab,cb->ac
            dX = self.ops.gemm(dY, self.W, trans2=True)
            # ac,ab->cb
            self.ops.gemm(X, dY, out=self.dW, trans1=True) 
            return dX
        return (queries, keys, values), backprop_get_inputs

    def compare(self, queries, keys, lengths):
        '''Compare queries and keys according to scaled dot product attention.
        
        Queries and keys should be equally-shaped ragged arrays, representing
        variable length sequences. Typically each row will represent a word.

        We return a ragged matrix which if padded would be of shape:
        (sum(lengths), window_size)
        '''
        (dotprod, dotprod_lengths), backprop_rwd = _ragged_window_dot(
            self.ops, queries, keys, lengths, self.nW)
        dotprod /= ops.xp.sqrt(self.nK)
        attention = self.ops.softmax_sequences(dotprod, dotprod_lengths)
        def backprop_attention(d_attention):
            d_dotprod = self.ops.backprop_softmax_sequences(d_attention,
                                                            attention,
                                                            dotprod_lengths)
            d_dotprod /= ops.xp.sqrt(self.nO)
            d_queries, d_keys = _backprop_rwd(d_dotprod)
            return d_queries, d_keys
        return (attention, dotprod_lengths), backprop_attention

    def rescale(self, V, A, window_sizes):
        '''Perform a weighted sum of values with the attention. Both arrays are ragged.'''
        raise NotImplementedError
        #output = self.ops.allocate(V.shape)
        #for i, window_size in enumerate(window_sizes):
        #    # TODO: This is wrong -- it doesn't center the window correctly =/
        #    values = V[start : start + window_size]
        #    attention = A[start : start + window_size]
        #    output[i] = (v_window * attention).sum(axis=0)
        #return output, backprop_rescale


def _ragged_window_dot(ops, X, Y, lengths, nW):
    '''Multiply X against context windows of Y, where X and Y are both ragged
    matrices, representing concatenated sequences. We output a ragged array
    where each entry is a vector with the dot product of X[i] against the
    context-window of Y, where the context-window is of variable length'''
    start = 0
    output = []
    for i, length in enumerate(lengths):
        X_ = X[start : start+length]
        Y_ = Y[start : start+length]
        for j in range(length):
            dots = X_[j].dot(Y_[max(j-nW, 0) : j + nW].T)
            output.append(dots)
        start += length
    # We do have to output a ragged array here...If we pad, it'll be frustrating
    # later, because our softmax will be off due to the padding.
    out_lengths = ops.asarray([len(out) for out in output])
    shape = tuple(X.shape)
    def backprop_ragged_window_dot(d_output):
        # Imagine we had the simple thing (without the raggedness, window):
        # output = einsum('nd,nd->nn', X, Y)
        # dX     = einsum('nn,nd->nd', d_output, Y)
        # dY     = einsum('nn,nd->nd', d_output, X)
        # Let w be the window size, and imagine we expanded and padded. Then:
        # output = einsum('nd,nwd->nw', X, window_Y)
        # dX     = einsum('nw,nwd->nd', d_output, window_Y)
        # dY     = einsum('nw,nwd->nd', d_output, window_X)
        dX = ops.allocate(shape)
        dY = ops.allocate(shape)
        d_output_list = ops.unflatten(d_output, out_lengths)
        for i, d_dots in enumerate(d_output_list):
            dX_ = X[start : start+length]
            dY_ = Y[start : start+length]
            for j in range(length):
                dX_[j] = TODO(Y_[max(j-nW, 0) : j + nW], d_dots)
                dY_[max(j-nW, 0) : j+nW] = TODO(X[j], d_dots)
    return ops.flatten(output), out_lengths


@describe.attributes(
    nO=Dimension("Output size"),
    Q=Synapses("Learned 'query' vector",
        lambda obj: (obj.nO, 1),
        lambda Q, ops: ops.normal_init(Q, Q.shape[0])),
    dQ=Gradient("Q"),
)
class ParametricAttention(Model):
    """Weight inputs by similarity to a learned vector"""
    name = 'para-attn'

    def __init__(self, nO=None, hard=False, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.hard = hard
        self.drop_factor = kwargs.get('drop_factor', 1.0)

    def begin_update(self, Xs_lengths, drop=0.):
        Xs, lengths = Xs_lengths
        attention, bp_attention = self._get_attention(self.Q, Xs, lengths)
        output, bp_output = self._apply_attention(attention, Xs, lengths)

        def attention_bwd(d_output, sgd=None):
            dXs, d_attention = bp_output(d_output)
            dQ, dXs2 = bp_attention(d_attention)
            self.dQ += dQ
            dXs += dXs2
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return dXs
        return (output, lengths), attention_bwd

    def _get_attention(self, Q, Xs, lengths):
        attention = Xs.dot(Q)
        if self.hard:
            start = 0
            for i, length in enumerate(lengths):
                argmax = attention[start:start+length].argmax()
                attention[start:start+length] = 0
                attention[start+argmax] = 1.
                start += length
        else:
            attention = self.ops.softmax_sequences(attention, lengths)
        def get_attention_bwd(d_attention):
            if self.hard:
                d_attention *= attention
            else:
                d_attention = self.ops.backprop_softmax_sequences(d_attention,
                                                                 attention,
                                                                 lengths)
            dQ = self.ops.gemm(Xs, d_attention, trans1=True)
            dXs = self.ops.xp.outer(d_attention, Q)
            return dQ, dXs
        return attention, get_attention_bwd

    def _apply_attention(self, attention, Xs, lengths):
        output = Xs * attention
        def apply_attention_bwd(d_output):
            d_attention = (Xs * d_output).sum(axis=1, keepdims=True)
            dXs         = d_output * attention
            return dXs, d_attention
        return output, apply_attention_bwd
