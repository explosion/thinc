# coding: utf8
from __future__ import unicode_literals

from .neural.pooling import Pooling, max_pool, mean_pool, sum_pool  # noqa: F401
from ._registry import registry


@registry.layers.register("MeanPooling.v1")
def make_MeanPooling_v1():
    return Pooling(mean_pool)


@registry.layers.register("SumPooling.v1")
def make_SumPooling_v1():
    return Pooling(sum_pool)


@registry.layers.register("MaxPooling.v1")
def make_MaxPooling_v1():
    return Pooling(max_pool)


@registry.layers.register("MeanMaxPooling.v1")
def make_MeanMaxPooling_v1():
    return Pooling(mean_pool, max_pool)
