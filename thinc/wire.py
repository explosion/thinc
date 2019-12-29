import copy

from .neural._classes.function_layer import FunctionLayer
from .neural._classes.function_layer import wrap, AdditionLayer, ConcatenationLayer
from .layers.feed_forward import FeedForward
from .layers.base import Model
from .util import is_ragged


def layerize(begin_update=None, predict=None, *args, **kwargs):
    """Wrap a function into a layer"""
    if begin_update is not None:
        return FunctionLayer(begin_update, predict=predict, *args, **kwargs)

    def wrapper(begin_update):
        return FunctionLayer(begin_update, *args, **kwargs)

    return wrapper


def noop(*layers):
    """Transform a sequences of layers into a null operation."""

    def noop_forward(X):
        return X, lambda D, *a, **k: D

    return layerize(noop_forward, layers=list(layers))


def create_variadic(layers, *, cls=None, function=None, **kwargs):
    """Create a layer for a variadic function, i.e. a function that can apply
    over a variable number of child layers. If the first child layer is already
    set up for the function, we just extend its children instead of creating
    a new layer.

    For instance, let's say we're concatenating a sequence of layers, defined
    using the | operator:

        layer = (child1 | child2 | child2)

    This will result in two calls:

        concatenate(concatenate(child1, child2), child3)

    With create_variadic, this will be flattened to:

        concatenate(child1, child2, child3)

    Which will be more efficient.
    """
    if not layers:
        return noop()
    elif cls is not None:
        if isinstance(layers[0], cls):
            main_layer = layers[0]
            others = layers[1:]
        elif isinstance(layers[-1], cls):
            main_layer = layers[0]
            others = layers[1:]
        else:
            return cls(layers=layers, **kwargs)
    elif function is not None:
        if layers[0].begin_update is function:
            main_layer = layers[0]
            others = layers[1:]
        elif layers[-1].begin_update is function:
            main_layer = layers[-1]
            others = layers[:-1]
        else:
            return FunctionLayer(function, layers=layers, **kwargs)
    else:
        raise ValueError("One of 'cls' or 'function' must be provided")
    for layer in others:
        main_layer.add_layer(layer)
    return main_layer


def chain(*layers):
    """Compose two models `f` and `g` such that they become layers of a single
    feed-forward model that computes `g(f(x))`.
    """
    return create_variadic(layers, cls=FeedForward)


def clone(orig, n):
    """Construct `n` copies of a layer, with distinct weights.

    i.e. `clone(f, 3)(x)` computes `f(f'(f''(x)))`.
    """
    if n == 0:
        return noop()
    layers = [orig]
    for i in range(n - 1):
        layers.append(copy.deepcopy(orig))
        layers[-1].set_id()
    return chain(*layers)


def concatenate(*layers):
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`
    """
    return create_variadic(layers, cls=ConcatenationLayer)


def add(*layers):
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    added, i.e. `add(f, g)(x)` computes `f(x) + g(x)`
    """
    return create_variadic(layers, cls=AdditionLayer)


@layerize
def flatten_add_lengths(seqs):
    """Transform sequences to ragged arrays if necessary. If sequences are
    already ragged, do nothing. A ragged array is a tuple (data, lengths),
    where data is the concatenated data.
    """
    if is_ragged(seqs):
        return seqs, lambda d_seqs: d_seqs

    ops = Model.ops
    lengths = ops.asarray([len(seq) for seq in seqs], dtype="i")

    def finish_update(d_X):
        return ops.unflatten(d_X, lengths)

    return (ops.flatten(seqs), lengths), finish_update


@layerize
def unflatten(X_lengths):
    """Transform sequences from a ragged format into lists."""
    ops = Model.ops
    X, lengths = X_lengths
    Xs = ops.unflatten(X, lengths)

    def backprop_unflatten(dXs):
        dX = ops.flatten(dXs, pad=0)
        return dX

    return Xs, backprop_unflatten


def with_getitem(idx, layer):
    """Transform data on the way into and out of a layer, by plucking an item
    from a tuple.
    """

    def with_getitem_forward(items):
        X, finish = layer.begin_update(items[idx])
        return items[:idx] + (X,) + items[idx + 1 :], finish

    return wrap(with_getitem_forward, layer)


def with_square_sequences(model):
    def padded_forward(seqs_in):
        padded_in, _, unpad = model.ops.square_sequences(seqs_in)
        (padded_out, _), backprop_model = model.begin_update(padded_in)
        seqs_out = unpad(padded_out)

        def backprop_padding(d_seqs_out):
            d_padded_out, sizes_at_t, unpad = model.ops.square_sequences(d_seqs_out)
            d_padded_in = backprop_model((d_padded_out, None))
            return unpad(d_padded_in)

        return seqs_out, backprop_padding

    return wrap(padded_forward, model)


def with_pad_and_mask(layer):
    """Wrap a layer so that list inputs are transformed into padded batches.
    The inputs are provided as (data, mask) tuples.
    """

    def create_model_input_forward(Xs):
        nX = layer.ops.asarray([x.shape[0] for x in Xs], dtype="i")
        nL = nX.max()
        X, unpad_X = layer.ops.pad_sequences(Xs, pad_to=nL)
        X_mask = _get_mask(layer.ops, X, nX)
        Y, bp_Y = layer.begin_update((X.astype("float32"), X_mask))

        def create_model_input_backward(dYs):
            dY, _ = layer.ops.pad_sequences(dYs, pad_to=nL)
            dX = bp_Y(dY)
            return unpad_X(dX)

        return unpad_X(Y), create_model_input_backward

    return wrap(create_model_input_forward, layer)


def _get_mask(ops, X, nX):
    nB = X.shape[0]
    nL = X.shape[1]
    X_mask = ops.allocate((nB, nL, nL))
    for i, length in enumerate(nX):
        X_mask[i, :, :length] = 1.0
    return X_mask
