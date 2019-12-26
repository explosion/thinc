import copy
import numpy

from .neural._classes.model import Model
from .neural._classes.feed_forward import FeedForward
from .neural._classes.function_layer import FunctionLayer
from .neural._classes.function_layer import wrap, AdditionLayer, ConcatenationLayer
from .neural.util import is_ragged


def layerize(begin_update=None, predict=None, *args, **kwargs):
    """Wrap a function into a layer"""
    if begin_update is not None:
        return FunctionLayer(begin_update, predict=predict, *args, **kwargs)

    def wrapper(begin_update):
        return FunctionLayer(begin_update, *args, **kwargs)

    return wrapper


def noop(*layers):
    """Transform a sequences of layers into a null operation."""

    def noop_forward(X, drop=0.0):
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
def flatten_add_lengths(seqs, drop=0.0):
    """Transform sequences to ragged arrays if necessary. If sequences are
    already ragged, do nothing. A ragged array is a tuple (data, lengths),
    where data is the concatenated data.
    """
    if is_ragged(seqs):
        return seqs, lambda d_seqs, sgd=None: d_seqs

    ops = Model.ops
    lengths = ops.asarray([len(seq) for seq in seqs], dtype="i")

    def finish_update(d_X, sgd=None):
        return ops.unflatten(d_X, lengths)

    return (ops.flatten(seqs), lengths), finish_update


@layerize
def unflatten(X_lengths, drop=0.0):
    """Transform sequences from a ragged format into lists."""
    ops = Model.ops
    X, lengths = X_lengths
    Xs = ops.unflatten(X, lengths)

    def backprop_unflatten(dXs, sgd=None):
        dX = ops.flatten(dXs, pad=0)
        return dX

    return Xs, backprop_unflatten


def with_reshape(layer):
    """Reshape data on the way into and out from a layer."""

    def with_reshape_forward(X, drop=0.0):
        initial_shape = X.shape
        final_shape = list(initial_shape[:-1]) + [layer.nO]
        nB = X.shape[0]
        nT = X.shape[1]
        X2d = X.reshape(-1, X.shape[2])
        X2d = X2d.astype(layer.ops.xp.float32)
        Y2d, Y2d_backprop = layer.begin_update(X2d, drop=drop)
        Y = Y2d.reshape(final_shape)

        def with_reshape_backward(dY, sgd=None):
            dY = dY.reshape(nB * nT, -1).astype(layer.ops.xp.float32)
            return Y2d_backprop(dY, sgd=sgd).reshape(initial_shape)

        return Y, with_reshape_backward

    return wrap(with_reshape_forward, layer)


def with_getitem(idx, layer):
    """Transform data on the way into and out of a layer, by plucking an item
    from a tuple. 
    """

    def with_getitem_forward(items, drop=0.0):
        X, finish = layer.begin_update(items[idx], drop=drop)
        return items[:idx] + (X,) + items[idx + 1 :], finish

    return wrap(with_getitem_forward, layer)


def with_square_sequences(model):
    def padded_forward(seqs_in, drop=0.0):
        padded_in, _, unpad = model.ops.square_sequences(seqs_in)
        (padded_out, _), backprop_model = model.begin_update(padded_in, drop=drop)
        seqs_out = unpad(padded_out)

        def backprop_padding(d_seqs_out, sgd=None):
            d_padded_out, sizes_at_t, unpad = model.ops.square_sequences(d_seqs_out)
            d_padded_in = backprop_model((d_padded_out, None), sgd=sgd)
            return unpad(d_padded_in)

        return seqs_out, backprop_padding

    return wrap(padded_forward, model)


def with_flatten(layer, pad=0, ndim=4):
    def with_flatten_forward(seqs_in, drop=0.0):
        lengths = layer.ops.asarray([len(seq) for seq in seqs_in])
        X, bp_layer = layer.begin_update(layer.ops.flatten(seqs_in, pad=pad), drop=drop)
        if bp_layer is None:
            return layer.ops.unflatten(X, lengths, pad=pad), None

        def finish_update(d_seqs_out, sgd=None):
            d_X = bp_layer(layer.ops.flatten(d_seqs_out, pad=pad), sgd=sgd)
            if d_X is None:
                return None
            else:
                return layer.ops.unflatten(d_X, lengths, pad=pad)

        return layer.ops.unflatten(X, lengths, pad=pad), finish_update

    def with_flatten_predict(seqs_in):
        lengths = layer.ops.asarray([len(seq) for seq in seqs_in])
        X = layer.predict(layer.ops.flatten(seqs_in, pad=pad))
        return layer.ops.unflatten(X, lengths, pad=pad)

    return wrap(
        with_flatten_forward,
        layer,
        predict=with_flatten_predict,
        name=f"with_flatten-{layer.name}",
        on_data_hooks=[_with_flatten_on_data],
    )


def _with_flatten_on_data(model, X, Y):
    X = model.ops.flatten(X)
    for layer in model._layers:
        for hook in layer.on_data_hooks:
            hook(layer, X, Y)
        X = layer(X)


def uniqued(layer, column=0):
    """Group inputs to a layer, so that the layer only has to compute
    for the unique values. The data is transformed back before output, and the same
    transformation is applied for the gradient. Effectively, this is a cache
    local to each minibatch.

    The uniqued wrapper is useful for word inputs, because common words are
    seen often, but we may want to compute complicated features for the words,
    using e.g. character LSTM.
    """

    def uniqued_fwd(X, drop=0.0):
        keys = X[:, column]
        keys = layer.ops.xp.ascontiguousarray(keys)
        if not isinstance(keys, numpy.ndarray):
            keys = keys.get()
        uniq_keys, ind, inv, counts = numpy.unique(
            keys, return_index=True, return_inverse=True, return_counts=True
        )
        X_uniq = layer.ops.xp.ascontiguousarray(X[ind])
        Y_uniq, bp_Y_uniq = layer.begin_update(X_uniq, drop=drop)
        Y = Y_uniq[inv].reshape((X.shape[0],) + Y_uniq.shape[1:])

        def uniqued_bwd(dY, sgd=None):
            dY_uniq = layer.ops.allocate(Y_uniq.shape, dtype="f")
            layer.ops.scatter_add(dY_uniq, layer.ops.asarray(inv, dtype="i"), dY)
            d_uniques = bp_Y_uniq(dY_uniq, sgd=sgd)
            if d_uniques is not None:
                dX = (d_uniques / counts)[inv]
                return dX
            else:
                return None

        return Y, uniqued_bwd

    model = wrap(uniqued_fwd, layer)
    return model


def foreach(layer, drop_factor=1.0):
    """Map a layer across list items"""

    def foreach_fwd(docs, drop=0.0):
        sents = []
        lengths = []
        for doc in docs:
            doc_sents = [sent for sent in doc if len(sent)]
            assert len(doc_sents)
            if drop:
                subset = [
                    s for s in doc_sents if numpy.random.random() >= drop * drop_factor
                ]
            else:
                subset = list(doc_sents)
            if subset:
                sents.extend(subset)
                lengths.append(len(subset))
            else:
                numpy.random.shuffle(doc_sents)
                sents.append(doc_sents[0])
                lengths.append(1)
        assert len(sents)
        flat, bp_flat = layer.begin_update(sents, drop=0.0)
        output = layer.ops.unflatten(flat, lengths)

        def foreach_bwd(d_output, sgd=None):
            d_flat = layer.ops.flatten(d_output)
            d_sents = bp_flat(d_flat, sgd=sgd)
            if d_sents is None:
                return d_sents
            else:
                return layer.ops.unflatten(d_sents, lengths)

        return output, foreach_bwd

    model = wrap(foreach_fwd, layer)

    def _run_foreach_child_hooks(model, X, y):
        for layer in model._layers:
            for hook in layer.on_data_hooks:
                hook(layer, X[0], y[0])

    model.on_data_hooks = [_run_foreach_child_hooks]

    return model


def foreach_sentence(layer, get_sents=None, drop_factor=1.0):
    """Map a layer across sentences"""
    if get_sents is None:
        get_sents = lambda doc: doc.sents

    def sentence_fwd(docs, drop=0.0):
        sents = []
        lengths = []
        for doc in docs:
            doc_sents = [sent for sent in get_sents(doc) if len(sent)]
            subset = [
                s for s in doc_sents if numpy.random.random() >= drop * drop_factor
            ]
            if subset:
                sents.extend(subset)
                lengths.append(len(subset))
            else:
                numpy.random.shuffle(doc_sents)
                sents.append(doc_sents[0])
                lengths.append(1)
        flat, bp_flat = layer.begin_update(sents, drop=0.0)
        output = layer.ops.unflatten(flat, lengths)

        def sentence_bwd(d_output, sgd=None):
            d_flat = layer.ops.flatten(d_output)
            d_sents = bp_flat(d_flat, sgd=sgd)
            if d_sents is None:
                return d_sents
            else:
                return layer.ops.unflatten(d_sents, lengths)

        return output, sentence_bwd

    return wrap(sentence_fwd, layer)


def with_pad_and_mask(layer):
    """Wrap a layer so that list inputs are transformed into padded batches.
    The inputs are provided as (data, mask) tuples.
    """
    def create_model_input_forward(Xs, drop=0.0):
        nX = layer.ops.asarray([x.shape[0] for x in Xs], dtype="i")
        nL = nX.max()
        X, unpad_X = layer.ops.pad_sequences(Xs, pad_to=nL)
        X_mask = _get_mask(layer.ops, X, nX)
        Y, bp_Y = layer.begin_update((X.astype("float32"), X_mask), drop=drop)

        def create_model_input_backward(dYs, sgd=None):
            dY, _ = layer.ops.pad_sequences(dYs, pad_to=nL)
            dX = bp_Y(dY, sgd=sgd)
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
