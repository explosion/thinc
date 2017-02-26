from ..api import layerize
import numpy

try:
    from cupy import  get_array_module
except ImportError:
    get_array_module = lambda arr: numpy


def Pooling(*funcs, **kwargs):
    ops = kwargs['ops'] if 'ops' in kwargs else funcs[0].ops
    F = len(funcs)
    def begin_update(X_lengths, drop=0.0):
        X, lengths = X_lengths
        X, bp_dropout = ops.dropout(X, drop)
        B, O = X.shape
        pooled = ops.allocate((F, len(lengths), O))
        bp_funcs = [None] * F
        for i, func in enumerate(funcs):
            pooled[i], bp_funcs[i] = func.begin_update((X, lengths))
        def finish_update(d_pooled, sgd=None):
            d_pooled = d_pooled.reshape((len(lengths), F, O))
            d_pooled = d_pooled.transpose((1, 0, 2))
            dX = ops.allocate(X.shape)
            for i, bp_func in enumerate(bp_funcs):
                dX += bp_func(d_pooled[i])
            return dX
        pooled = pooled.transpose((1, 0, 2))
        pooled = pooled.reshape((len(lengths), F * O))
        return pooled, finish_update
    return layerize(begin_update)


@layerize
def mean_pool(X_lengths, drop=0.):
    X, lengths = X_lengths
    xp = get_array_module(X)
    output = xp.zeros((len(lengths), X.shape[1]), dtype='float32')
    start = 0
    for i, length in enumerate(lengths):
        end = start + length
        output[i] = X[start : end].mean(axis=0)
        start = end
    def finish_update(d_output, sgd=None):
        d_X = xp.zeros((X.shape[0], X.shape[1]), dtype='float32')
        start = 0
        for i, length in enumerate(lengths):
            end = start + length
            d_X[start : end] += d_output[i] / (end-start)
            start = end
        return d_X
    return output, finish_update


@layerize
def max_pool(X_lengths, drop=0.):
    X, lengths = X_lengths
    xp = get_array_module(X)
    maxes = xp.zeros((len(lengths), X.shape[1]), dtype='float32')
    start = 0
    for i, length in enumerate(lengths):
        end = start + length
        maxes[i] = X[start : end].max(axis=0)
        start = end
    def finish_update(d_maxes, sgd=None):
        d_X = xp.zeros((X.shape[0], X.shape[1]), dtype='float32')
        start = 0
        for i, length in enumerate(lengths):
            end = start + length
            d_X[start : end] += d_maxes[i] * (d_maxes[i] == maxes[i])
            start = end
        return d_X
    return maxes, finish_update
