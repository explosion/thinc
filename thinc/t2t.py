# coding: utf8
from __future__ import unicode_literals

from .neural._classes.convolution import ExtractWindow  # noqa: F401
from .neural._classes.attention import ParametricAttention  # noqa: F401
from .neural._classes.rnn import LSTM, BiLSTM  # noqa: F401
from .neural._classes.multiheaded_attention import MultiHeadedAttention
from .neural._classes.multiheaded_attention import prepare_self_attention
from ._registry import registry


@registry.layers.register("ExtractWindow.v1")
def make_ExtractWindow(window_size):
    return ExtractWindow(nW=window_size)


@registry.layers.register("ParametricAttention.v1")
def make_ParametricAttention(outputs):
    return ParametricAttention(nO=outputs)
