# coding: utf8
from __future__ import unicode_literals

from .neural._classes.convolution import ExtractWindow  # noqa: F401
from .neural._classes.attention import ParametricAttention  # noqa: F401
from .neural._classes.rnn import LSTM, BiLSTM  # noqa: F401
from .neural._classes.multiheaded_attention import MultiHeadedAttention
from .neural._classes.multiheaded_attention import prepare_self_attention


def SelfAttention(width, depth, pieces=4):
    from .api import chain, clone, with_getitem
    from .misc import LayerNorm, Residual
    from .v2v import Mish, Affine

    if (width % pieces) != 0:
        raise ValueError("Width must be divisible by pieces (aka heads)")
    return clone(
        Residual(
            chain(
                prepare_self_attention(
                    Affine(width*3, width),
                    nM=width, nH=pieces),
                MultiHeadedAttention(),
                with_getitem(0, LayerNorm(Mish(width, width))),
            )
        ),
        depth
    )
