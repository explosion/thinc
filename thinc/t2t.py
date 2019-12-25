# coding: utf8
from __future__ import unicode_literals

from .neural._classes.convolution import ExtractWindow  # noqa: F401
from .neural._classes.attention import ParametricAttention  # noqa: F401
from .neural._classes.rnn import LSTM, BiLSTM  # noqa: F401
from .neural._classes.multiheaded_attention import MultiHeadedAttention
from .neural._classes.multiheaded_attention import prepare_self_attention
from ._registry import registry
from .api import layerize, noop, with_square_sequences
from .extra.wrappers import PyTorchWrapperRNN


@registry.layers.register("ExtractWindow.v1")
def make_ExtractWindow(window_size: int):
    return ExtractWindow(nW=window_size)


@registry.layers.register("ParametricAttention.v1")
def make_ParametricAttention(outputs: int):
    return ParametricAttention(nO=outputs)


@registry.layers.register("TorchBiLSTM.v1")
def make_TorchBiLSTM(outputs: int, inputs: int, depth: int, dropout: float=0.2):
    import torch.nn
    if depth == 0:
        return layerize(noop())
    model = torch.nn.LSTM(nI, nO // 2, depth, bidirectional=True, dropout=dropout)
    return with_square_sequences(PyTorchWrapperRNN(model))


@registry.layers.register("MaxoutWindowEncoder.v1")
def make_MaxoutWindowEncoder(width: int, depth: int, *, pieces: int, window_size: int):
    from .neural._classes.model import Model
    from .neural._classes.maxout import Maxout
    from .neural._classes.resnet import Residual
    from .neural._classes.layernorm import LayerNorm
    from .api import chain, clone

    n_tokens = (window_size * 2) + 1
    with Model.define_operators({">>": chain, "**": clone}):
        model = Residual(
            ExtractWindow(nW=window)
            >> LayerNorm(Maxout(width, width * n_tokens, pieces=pieces))
        ) ** depth
    model.nO = nO
    model.receptive_field = n_tokens * depth
    return model


@registry.layers.register("MishWindowEncoder.v1")
def make_MishWindowEncoder(width: int, depth: int, *, window_size: int):
    from .neural._classes.mish import Mish
    from .neural._classes.model import Model
    from .neural._classes.resnet import Residual
    from .neural._classes.layernorm import LayerNorm
    from .api import chain, clone

    n_tokens = (window_size * 2) + 1
    with Model.define_operators({">>": chain, "**": clone}):
        model = Residual(
            ExtractWindow(nW=window)
            >> LayerNorm(Mish(width, width * n_tokens))
        ) ** depth
    model.nO = nO
    model.receptive_field = n_tokens * depth
    return model
