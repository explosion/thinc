from ..api import layerize
import numpy

from .ops import NumpyOps, CupyOps

from ._classes.model import Model


try:
    from cupy import  get_array_module
except ImportError:
    get_array_module = lambda arr: numpy


def Pooling(*funcs, **kwargs):
    ops = kwargs['ops'] if 'ops' in kwargs else Model.ops
    F = len(funcs)
    drop_factor = kwargs.get('drop_factor', 1.0)
    def begin_update(X_lengths, drop=0.0):
        drop *= drop_factor
        X, lengths = X_lengths
        T, O = X.shape
        pooled = ops.allocate((len(lengths), F*O))
        bp_funcs = [None] * F
        for i, func in enumerate(funcs):
            res, bp_res = func.begin_update((X, lengths))
            pooled[:, i*O:i*O+O] = res
            bp_funcs[i] = bp_res
        pooled, bp_dropout = ops.dropout(pooled, drop)
        def finish_update(d_pooled, sgd=None):
            dX = ops.allocate(X.shape)
            for i, bp_func in enumerate(bp_funcs):
                dX += bp_func(d_pooled[:, i*O : i*O+O])
            return dX
        return pooled, bp_dropout(finish_update)
    return layerize(begin_update)


@layerize
def mean_pool(X_lengths, drop=0.):
    X, lengths = X_lengths
    if isinstance(X, numpy.ndarray):
        ops = NumpyOps()
    else:
        ops = CupyOps()
    output = ops.mean_pool(X, lengths)
    def finish_update(d_output, sgd=None):
        d_output = ops.xp.ascontiguousarray(d_output)
        return ops.backprop_mean_pool(d_output, lengths)
    return output, finish_update


@layerize
def sum_pool(X_lengths, drop=0.):
    X, lengths = X_lengths
    if isinstance(X, numpy.ndarray):
        ops = NumpyOps()
    else:
        ops = CupyOps()

    output = ops.sum_pool(X, lengths)
    def finish_update(d_output, sgd=None):
        d_output = ops.xp.ascontiguousarray(d_output)
        return ops.backprop_sum_pool(d_output, lengths)
    return output, finish_update


@layerize
def max_pool(X_lengths, drop=0.):
    X, lengths = X_lengths
    ops = Model.ops

    best, which = ops.max_pool(X, lengths)
    def finish_update(d_output, sgd=None):
        d_output = ops.xp.ascontiguousarray(d_output)
        return ops.backprop_max_pool(d_output, which, lengths)
    return best, finish_update
