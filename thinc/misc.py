# coding: utf8
from __future__ import unicode_literals

from .neural._classes.batchnorm import BatchNorm  # noqa: F401
from .neural._classes.layernorm import LayerNorm  # noqa: F401
from .neural._classes.resnet import Residual  # noqa: F401
from .neural._classes.feature_extracter import FeatureExtracter  # noqa: F401
from .neural._classes.function_layer import FunctionLayer  # noqa: F401
from .neural._classes.feed_forward import FeedForward  # noqa: F401
from ._registry import registry


@registry.layers.register("FeatureExtractor.v1")
def make_FeatureExtractor(attrs):
    return FeatureExtracter(attrs)


@registry.layers.register("LayerNorm.v1")
def make_LayerNorm(outputs=None, child=None):
    return LayerNorm(nO=outputs, child=child)
