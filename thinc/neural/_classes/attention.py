from ... import describe
from ...describe import Dimension, Synapses, Gradient
from .model import Model


@describe.attributes(
    nO=Dimension("Output size"),
    Q=Synapses("Learned 'query' vector",
        lambda obj: (1, obj.nO),
        lambda Q, ops: Q.fill(1)),
    dQ=Gradient("Q"),
)
class ParametricAttention(Model):
    """Weight inputs by similarity to a learned vector"""
    name = 'para-attn'

    def __init__(self, nO=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
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
        attention = Xs * Q
        start = 0
        for i, length in enumerate(lengths):
            self.ops.softmax(attention[start : start+length], inplace=True)
            start += length
        def get_attention_bwd(d_attention):
            d_attention = backprop_softmax(d_attention, attention, lengths)
            dQ = (Xs * d_attention).sum(axis=0, keepdims=True)
            dXs = d_attention * Q
            return dQ, dXs
        return attention, get_attention_bwd

    def _apply_attention(self, attention, Xs, lengths):
        output = Xs * attention
        def apply_attention_bwd(d_output):
            d_attention = d_output * Xs
            dXs        = d_output * attention
            return dXs, d_attention
        return output, apply_attention_bwd
 

def backprop_softmax(dy, y, lengths):
    dx = y * dy
    start = 0
    for i, length in enumerate(lengths):
        sumdx = dx[start : start+length].sum(axis=1, keepdims=True)
        dx[start:start+length] -= y[start : start+length] * sumdx
        start += length
    return dx


