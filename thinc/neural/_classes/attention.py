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
        # Let's say we have three inputs, of lengths [10, 12, 3], d=128
        # X becomes (25, 128)
        (queries, keys, values, key_lengths), get_dX = self.get_inputs(X)
        # We attend over a window of 5 words previous, 5 words following.
        # We project queries and keys down to d=32, values to 64. This gives:
        # queries: (25, 32)
        # keys: (sum(kv_lengths), 32)
        # values: (sum(kv_lengths), 64)
        attention, backprop_attention = self.get_attention(queries, keys, key_lengths)
        # Attention table is something like (25, 10), except not all words
        # will have a full window. Instead the values are concatenated, and
        # we keep a lengths vector.
        # The weight attention[i, j] tells us about the pair (X[i], X[i+context[j]]
        # where context is [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        weighted = values * attention # (sum(kv_lengths), 54)
        output = self.ops.sum_pool(weighted, kv_lengths)

        def self_attention_finish_update(d_output, sgd=None):
            d_weighted = self.ops.backprop_sum_pool(d_output, kv_lengths)
            d_attention = d_weighted * values
            d_values    = d_weighted * attention
            d_queries, d_keys = backprop_attention(d_attention)
            dX = get_dX(d_queries, d_keys, d_values) if sgd is not None:
                sgd(self._mem.weights, self._mem.gradients, key=self._mem.key)
            return dX
        return output, self_attention_finish_update

    def get_inputs(self, X, lengths):
        # Let's say X is (25, 300)
        # If nK=32 and nO=64, we need to project down to 32+32+64=128
        # So we have a weight matrix of shape (300, 128)
        Y = self.ops.gemm(X, self.W) # (25, 128)
        left_lengths, right_lengths = get_kv_lengths(lengths, self.nL, self.nR)
        kv_lengths = left_lengths + right_lengths
        queries = Y[:, :self.nK] # Shape (25, 32)
        # This will be shape (sum(kv_lengths), 32)
        keys = concat_contexts(Y[:, self.nK:self.nK*2], left_lengths, right_lengths)
        # This will be shape (sum(kv_lengths), 64)
        values = concat_contexts(Y[:, self.nK*2:, left_lengths, right_lengths) 
        def backprop_get_inputs(d_queries, d_keys, d_values):
            dY = self.ops.xp.hstack((d_queries, d_keys, d_values)) # (25, 128)
            # ab,cb->ac
            dX = self.ops.gemm(dY, self.W, trans2=True)
            # ac,ab->cb
            self.ops.gemm(X, dY, out=self.dW, trans1=True) 
            return dX
        return (queries, keys, values, kv_lengths), backprop_get_inputs

    def get_attention(self, queries, keys, lengths):
        # TODO: Figure out how to do this
        dotprod = _ragged_dot(self.ops, queries, keys, lengths)
        # Should get back a vector of shape (N,), representing a ragged
        # matrix of shape (25, <10)
        dotprod /= ops.xp.sqrt(self.nK)
        attention = self.ops.softmax_sequences(dotprod, lengths)
        def backprop_attention(d_attention):
            d_dotprod = self.ops.backprop_softmax_sequences(d_attention,
                                                            attention,
                                                            lengths)
            d_dotprod /= ops.xp.sqrt(self.nO)
            # Keys has shape (N, 32), d_dotprod has shape (N, 32)
            d_queries = self.ops.gemm(keys, d_attention, trans1=True)
            d_keys = self.ops.gemm(queries, d_attention)
            return d_queries, d_keys
        return attention, backprop_attention


def get_kv_lengths(lengths, nL, nR):
    '''Calculate how much left context and how much right context is available
    for sequences.'''
    lefts = numpy.zeros(lengths.shape, dtype='i')
    rights = numpy.zeros(lengths.shape, dtype='i')
    for i, length in enumerate(lengths):
        lefts[i] = max(0, i-nL)
        rights[i] = min(length, i+nR)
    return lefts, rights


def concat_contexts(vectors, left_lengths, right_lengths):
    '''Create a ragged array with the surrounding contexts. Contexts can be
    variable length, if only part of the context is available.'''
    xp = get_array_module(vectors)
    n = left_lengths.sum() + right_lengths.sum()
    output = xp.zeros((n, vectors.shape[1]), dtype='f')
    idx = 0
    for i in range(vectors.shape[0]):
        nL = left_lengths[i]
        nR = right_lengths[i]
        assert nL <= i
        assert (i+nR) < vectors.shape[0]
        output[idx : idx + nL] = vectors[i-nL : i]
        idx += nL
        output[idx : idx + nR] = vectors[i+1 : i+nR]
        idx += nR
    return output


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


