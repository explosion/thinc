import copy

from .neural._classes.feed_forward import FeedForward
from .neural._classes.model import Model
from . import check
from .check import equal_axis
from . import describe


def layerize(begin_update=None, *args, **kwargs):
    '''Wrap a function into a layer'''
    if begin_update is not None:
        return FunctionLayer(begin_update, *args, **kwargs)
    def wrapper(begin_update):
        return FunctionLayer(begin_update, *args, **kwargs)
    return wrapper


def metalayerize(user_func):
    '''Wrap a function over a sequence of layers and an input into a layer.'''
    def returned(layers, *args, **kwargs):
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
        return FeedForward([])
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


def concatenate(*layers): # pragma: no cover
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
                if bwd is not None:
                    d = bwd(gradient[:, start : end], *args, **kwargs)
                    if d is not None:
                        layer_grads[-1] += d
                start = end
            if layer_grads:
                return ops.asarray(layer_grads[-1])
            else:
                return None
        return output, finish_update
    layer = FunctionLayer(begin_update)
    layer._layers = layers
    return layer

 
def add(layer1, layer2):
    def begin_update(X, drop=0.0):
        out1, upd1 = layer1.begin_update(X, drop=drop)
        out2, upd2 = layer2.begin_update(X, drop=drop)
        def finish_update(gradient, sgd=None):
            grad1 = None
            grad2 = None
            if upd2 is not None:
                grad2 = upd2(gradient)
            if upd1 is not None:
                grad1 = upd1(gradient)
            if grad1 is not None and grad2 is not None:
                return grad1 + grad2
            elif grad1 is not None:
                return grad1
            else:
                return grad2
        return out1 + out2, finish_update
    model = layerize(begin_update)
    model._layers = [layer1, layer2]
    return model


def split_backward(layers): # pragma: no cover
    '''Separate a sequence of layers' `begin_update` methods into two lists of
    functions: one that computes the forward values, and the other that completes
    the backward pass. The backward sequence is only populated after the forward
    functions have been applied.
    '''
    backward = []
    forward = [sink_return(op.begin_update, backward.append)
               for op in layers]
    return forward, backward


def sink_return(func, sink, splitter=None): # pragma: no cover
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


def Arg(i):
    @layerize
    def begin_update(batched_inputs, drop=0.):
        inputs = zip(*batched_inputs)
        return inputs[i], None
    return begin_update


def with_flatten(layer):
    def begin_update(seqs_in, drop=0.):
        lengths = [len(seq) for seq in seqs_in]
        X, bp_layer = layer.begin_update(layer.ops.flatten(seqs_in), drop=drop)
        if bp_layer is None:
            return layer.ops.unflatten(X, lengths), None

        def finish_update(d_seqs_out, sgd=None):
            d_X = bp_layer(layer.ops.flatten(d_seqs_out), sgd=sgd)
            return layer.ops.unflatten(d_X, lengths) if d_X is not None else None
        return layer.ops.unflatten(X, lengths), finish_update
    model = layerize(begin_update)
    model._layers.append(layer)
    model.on_data_hooks.append(_with_flatten_on_data)
    model.name = 'flatten'
    return model

def _with_flatten_on_data(model, X, y):
    X = model.ops.flatten(X)
    for layer in model._layers:
        for hook in layer.on_data_hooks:
            hook(layer, X, y)
        X = layer(X)


class FunctionLayer(Model):
    '''Wrap functions into weightless Model instances, for use as network
    components.'''
    def __init__(self, begin_update, predict=None, predict_one=None,
            nI=None, nO=None, *args, **kwargs):
        self.begin_update = begin_update
        if predict is not None:
            self.predict = predict
        if predict_one is not None:
            self.predict_one = predict_one
        self.nI = nI
        self.nO = nO
        Model.__init__(self)
