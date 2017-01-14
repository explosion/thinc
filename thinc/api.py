import copy

from .neural._classes.feed_forward import FeedForward
from .neural._classes.model import Model


def layerize(begin_update=None, *args, **kwargs):
    '''Wrap a function into a layer'''
    if begin_update is not None:
        return FunctionLayer(begin_update, *args, **kwargs)
    def wrapper(begin_update):
        return FunctionLayer(begin_update, *args, **kwargs)
    return wrapper


def metalayerize(user_func):
    '''Wrap a function over a sequence of layers and an input into a layer.'''
    def returned(layers, X, *args, **kwargs):
        def begin_update(X, *args, **kwargs):
            return user_func(layers, X, *args, **kwargs)
        return FunctionLayer(begin_update, *args, **kwargs)
    return returned


def noop(*layers):
    '''Transform a sequences of layers into a null operation.'''
    def begin_update(X):
        return X, lambda D, *a, **k: D
    return begin_update


def chain(*layers):
    '''Compose two models `f` and `g` such that they become layers of a single
    feed-forward model that computes `g(f(x))`.
    
    Raises exception if their dimensions don't match.
    '''
    if len(layers) == 0:
        return FeedForward()
    elif len(layers) == 1:
        return layers[0]
    else:
        return FeedForward(layers)


def clone(orig, n):
    '''Construct `n` copies of a layer, with distinct weights.
    
    i.e. `clone(f, 3)(x)` computes `f(f'(f''(x)))`.
    '''
    layers = [orig]
    for i in range(n-1):
        layers.append(copy.deepcopy(orig))
    return FeedForward(layers)


def concatenate(*layers):
    '''Compose two or more models `f`, `g`, etc, such that their outputs are
    concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`
    '''
    if not layers:
        return noop()
    ops = layers[0].ops
    def begin_update(X, *a, **k):
        forward, backward = split_backward(layers)
        values = [fwd(X, *a, **k) for fwd in forward]
       
        output = ops.xp.hstack(values)
        shapes = [val.shape for val in values]

        def finish_update(gradient, *args, **kwargs):
            layer_grads = []
            start = 0
            for bwd, shape in zip(backward, shapes):
                end = start + shape[1]
                layer_grads.append(bwd(gradient[:, start : end], *args, **kwargs))
                start = end
            return ops.asarray(ops.xp.sum(layer_grads, axis=0))
        return output, finish_update
    layer = FunctionLayer(begin_update)
    return layer


def split_backward(layers):
    '''Separate a sequence of layers' `begin_update` methods into two lists of
    functions: one that computes the forward values, and the other that completes
    the backward pass. The backward sequence is only populated after the forward
    functions have been applied.
    '''
    backward = []
    forward = [sink_return(op.begin_update, backward.append)
               for op in layers]
    return forward, backward


def sink_return(func, sink, splitter=None):
    '''Transform a function `func` that returns tuples into a function that returns
    single values. Call a function `sink` on the unused values.
    '''
    def wrap(*args, **kwargs):
        output = func(*args, **kwargs)
        if splitter is None:
            to_keep, to_sink = output
        else:
            to_keep, to_sink = splitter(*output)
        sink(to_sink)
        return to_keep
    return wrap


class FunctionLayer(Model):
    '''Wrap functions into weightless Model instances, for use as network
    components.'''
    def __init__(self, begin_update, predict_batch=None, predict_one=None,
            nr_in=None, nr_out=None, *args, **kwargs):
        self.begin_update = begin_update
        self.predict_batch = predict_batch
        self.predict_one = predict_one
        self.nr_in = nr_in
        self.nr_out = nr_out

    def __call__(self, X):
        if self.predict_batch is not None:
            return self.predict_batch(X)
        else:
            X, _ = self.begin_update(X)
            return X

    def check_input(self, X, expect_batch=False):
        return True
