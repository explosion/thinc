from typing import Optional, Tuple
from ..model import Model
from ..types import Array
from ..util import get_width


def ParametricAttention(self, nO: Optional[int] = None):
    """Weight inputs by similarity to a learned vector"""
    return Model("para-attn", forward, init=init, params={"Q": None}, dims={"nO": nO})


def init(model, X=None, Y=None):
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    model.set_param("Q", model.ops.allocate((model.get_dim("nO"),)))


def forward(model, Xs_lengths: Tuple[Array, Array], is_train) -> Tuple[Array, Array]:
    Xs, lengths = Xs_lengths
    Q = model.get_param("Q")
    attention, bp_attention = _get_attention(model.ops, Q, Xs, lengths)
    output, bp_output = _apply_attention(model.ops, attention, Xs, lengths)

    def backprop(d_output_lengths):
        d_output, lengths = d_output_lengths
        dXs, d_attention = bp_output(d_output)
        dQ, dXs2 = bp_attention(d_attention)
        model.inc_grad("dQ", dQ)
        dXs += dXs2
        return (dXs, lengths)

    return (output, lengths), backprop


def _get_attention(ops, Q, Xs, lengths):
    attention = ops.gemm(Xs, Q.reshape((-1, 1)))
    attention = ops.softmax_sequences(attention, lengths)

    def get_attention_bwd(d_attention):
        d_attention = ops.backprop_softmax_sequences(d_attention, attention, lengths)
        dQ = ops.gemm(Xs, d_attention, trans1=True)
        dXs = ops.xp.outer(d_attention, Q)
        return dQ, dXs

    return attention, get_attention_bwd


def _apply_attention(self, attention, Xs, lengths):
    output = Xs * attention

    def apply_attention_bwd(d_output):
        d_attention = (Xs * d_output).sum(axis=1, keepdims=True)
        dXs = d_output * attention
        return dXs, d_attention

    return output, apply_attention_bwd
