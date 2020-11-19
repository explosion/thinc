# coding: utf8
from __future__ import unicode_literals

from ... import describe
from ...describe import Dimension, Synapses, Gradient
from .model import Model


@describe.attributes(
    nO=Dimension("Output size"),
    Q=Synapses(
        "Learned 'query' vector",
        lambda obj: (obj.nO, 1),
        lambda Q, ops: ops.normal_init(Q, Q.shape[0]),
    ),
    dQ=Gradient("Q"),
)
class ParametricAttention(Model):
    """Weight inputs by similarity to a learned vector"""

    name = "para-attn"

    def __init__(self, nO=None, hard=False, seed=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.hard = hard
        self.drop_factor = kwargs.get("drop_factor", 1.0)

        if seed is not None:
            self.seed = seed
        else:
            self.seed = self.id

    def begin_update(self, Xs_lengths, drop=0.0):
        Xs, lengths = Xs_lengths
        attention, bp_attention = self._get_attention(self.Q, Xs, lengths)
        output, bp_output = self._apply_attention(attention, Xs, lengths)

        def attention_bwd(d_output, sgd=None):
            dXs, d_attention = bp_output(d_output)
            dQ, dXs2 = bp_attention(d_attention)
            self.dQ += dQ
            dXs += dXs2
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.seed)
            return dXs

        return (output, lengths), attention_bwd

    def _get_attention(self, Q, Xs, lengths):
        attention = Xs.dot(Q)
        if self.hard:
            start = 0
            for i, length in enumerate(lengths):
                argmax = attention[start : start + length].argmax()
                attention[start : start + length] = 0
                attention[start + argmax] = 1.0
                start += length
        else:
            attention = self.ops.softmax_sequences(attention, lengths)

        def get_attention_bwd(d_attention):
            if self.hard:
                d_attention *= attention
            else:
                d_attention = self.ops.backprop_softmax_sequences(
                    d_attention, attention, lengths
                )
            dQ = self.ops.gemm(Xs, d_attention, trans1=True)
            dXs = self.ops.xp.outer(d_attention, Q)
            return dQ, dXs

        return attention, get_attention_bwd

    def _apply_attention(self, attention, Xs, lengths):
        output = Xs * attention

        def apply_attention_bwd(d_output):
            d_attention = (Xs * d_output).sum(axis=1, keepdims=True)
            dXs = d_output * attention
            return dXs, d_attention

        return output, apply_attention_bwd
