# coding: utf8
from __future__ import unicode_literals

import copy
import numpy

from .neural._classes.model import Model
from .neural._classes.function_layer import FunctionLayer
from .neural._classes.feed_forward import FeedForward


def layerize(begin_update=None, predict=None, *args, **kwargs):
    """Wrap a function into a layer"""
    if begin_update is not None:
        return FunctionLayer(begin_update, predict=predict, *args, **kwargs)

    def wrapper(begin_update):
        return FunctionLayer(begin_update, *args, **kwargs)

    return wrapper


def metalayerize(user_func):
    """Wrap a function over a sequence of layers and an input into a layer."""

    def returned(layers, *args, **kwargs):
        def begin_update(X, *args, **kwargs):
            return user_func(layers, X, *args, **kwargs)

        return FunctionLayer(begin_update, *args, **kwargs)

    return returned


@layerize
def flatten_add_lengths(seqs, pad=0, drop=0.0):
    ops = Model.ops
    lengths = ops.asarray([len(seq) for seq in seqs], dtype="i")

    def finish_update(d_X, sgd=None):
        return ops.unflatten(d_X, lengths, pad=pad)

    X = ops.flatten(seqs, pad=pad)
    return (X, lengths), finish_update

@layerize
def unflatten(X_lengths, drop=0.):
    ops = Model.ops
    X, lengths = X_lengths
    Xs = ops.unflatten(X, lengths)

    def backprop_unflatten(dXs, sgd=None):
        dX = ops.flatten(dXs, pad=0)
        return dX

    return Xs, backprop_unflatten

def remap_ids(ops=None, column=0):
    id_map = {0: 0}

    def remap_ids_fwd(ids, drop=0.0):
        ids = ids[:, column]
        if not isinstance(ids, numpy.ndarray):
            ids = ids.get()
        n_vector = len(id_map)
        for i, id_ in enumerate(ids):
            id_ = int(id_)
            if id_ not in id_map:
                id_map[id_] = n_vector
                n_vector += 1
            ids[i] = id_map[id_]
        return ops.asarray(ids), None

    model = layerize(remap_ids_fwd)
    if ops is None:
        ops = model.ops
    return model


def with_reshape(layer):
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
    def begin_update(items, drop=0.0):
        X, finish = layer.begin_update(items[idx], drop=drop)
        return items[:idx] + (X,) + items[idx + 1 :], finish

    model = layerize(begin_update)
    model._layers.append(layer)

    def on_data(self, items, y):
        for hook in layer.on_data_hooks:
            hook(layer, items[idx], y)

    model.on_data_hooks.append(on_data)
    return model


def noop(*layers):
    """Transform a sequences of layers into a null operation."""

    def begin_update(X, drop=0.0):
        return X, lambda D, *a, **k: D

    return begin_update


def chain(*layers):
    """Compose two models `f` and `g` such that they become layers of a single
    feed-forward model that computes `g(f(x))`.

    Raises exception if their dimensions don't match.
    """
    if len(layers) == 0:
        return FeedForward([])
    elif len(layers) == 1:
        return layers[0]
    else:
        return FeedForward(layers)


def clone(orig, n):
    """Construct `n` copies of a layer, with distinct weights.

    i.e. `clone(f, 3)(x)` computes `f(f'(f''(x)))`.
    """
    if n == 0:
        return layerize(noop())
    layers = [orig]
    for i in range(n - 1):
        layers.append(copy.deepcopy(orig))
        layers[-1].set_id()
    return FeedForward(layers)


def concatenate(*layers):  # pragma: no cover
    """Compose two or more models `f`, `g`, etc, such that their outputs are
    concatenated, i.e. `concatenate(f, g)(x)` computes `hstack(f(x), g(x))`
    """
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
                    d = bwd(
                        ops.xp.ascontiguousarray(gradient[:, start:end]),
                        *args,
                        **kwargs
                    )
                    if d is not None and hasattr(X, "shape"):
                        if not layer_grads:
                            layer_grads.append(d)
                        else:
                            layer_grads[-1] += d
                start = end
            if layer_grads:
                return ops.asarray(layer_grads[-1])
            else:
                return None

        return output, finish_update

    layer = FunctionLayer(begin_update)
    layer._layers = list(layers)

    def on_data(self, X, y=None):
        for layer in self._layers:
            for hook in layer.on_data_hooks:
                hook(layer, X, y)

    layer.on_data_hooks.append(on_data)
    return layer


def add(*layers):
    if not layers:
        return noop()

    def forward(X, drop=0.0):
        outs, callbacks = zip(*[lyr.begin_update(X, drop=drop) for lyr in layers])
        out = outs[0]
        for o in outs:
            out += o

        def backward(d_out, sgd=None):
            grads = [bp(d_out, sgd=sgd) for bp in callbacks if bp is not None]
            grads = [g for g in grads if g is not None]
            if grads:
                total = grads[0]
                for g in grads:
                    total += g
                return total
            else:
                return None

        return out, backward

    model = layerize(forward)
    model._layers = list(layers)

    def on_data(self, X, y):
        for layer in layers:
            for hook in layer.on_data_hooks:
                hook(layer, X, y)

    model.on_data_hooks.append(on_data)
    return model


def split_backward(layers):  # pragma: no cover
    """Separate a sequence of layers' `begin_update` methods into two lists of
    functions: one that computes the forward values, and the other that completes
    the backward pass. The backward sequence is only populated after the forward
    functions have been applied.
    """
    backward = []
    forward = [sink_return(op.begin_update, backward.append) for op in layers]
    return forward, backward


def sink_return(func, sink, splitter=None):  # pragma: no cover
    """Transform a function `func` that returns tuples into a function that returns
    single values. Call a function `sink` on the unused values.
    """

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
    def begin_update(batched_inputs, drop=0.0):
        inputs = list(zip(*batched_inputs))
        return inputs[i], None

    return begin_update


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
    def begin_update(seqs_in, drop=0.0):
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

    def predict(seqs_in):
        lengths = layer.ops.asarray([len(seq) for seq in seqs_in])
        X = layer(layer.ops.flatten(seqs_in, pad=pad))
        return layer.ops.unflatten(X, lengths, pad=pad)

    model = layerize(begin_update, predict=predict)
    model._layers.append(layer)
    model.on_data_hooks.append(_with_flatten_on_data)
    model.name = "flatten"
    return model


def _with_flatten_on_data(model, X, y):
    X = model.ops.flatten(X)
    for layer in model._layers:
        for hook in layer.on_data_hooks:
            hook(layer, X, y)
        X = layer(X)


def get_word_ids(ops, pad=1, token_drop=0.0, ignore=None):
    # TODO: Is this made obsolete by the FeatureExtractor?
    def forward(docs, drop=0.0):
        """Get word forms."""
        seqs = []
        ops = Model.ops
        for doc in docs:
            if ignore is not None:
                doc = [token for token in doc if not ignore(token)]
            # seq = [0] * pad
            seq = [(token.lex_id or token.orth) for token in doc]
            # seq += [0] * pad
            seqs.append(ops.asarray(seq, dtype="uint64"))
        return seqs, None

    return layerize(forward)


def wrap(func, *child_layers):
    model = layerize(func)
    model._layers.extend(child_layers)

    def on_data(self, X, y):
        for child in self._layers:
            for hook in child.on_data_hooks:
                hook(child, X, y)

    model.on_data_hooks.append(on_data)
    return model


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


def foreach_sentence(layer, drop_factor=1.0):
    """Map a layer across sentences (assumes spaCy-esque .sents interface)"""

    def sentence_fwd(docs, drop=0.0):
        sents = []
        lengths = []
        for doc in docs:
            doc_sents = [sent for sent in doc.sents if len(sent)]
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

    model = wrap(sentence_fwd, layer)
    return model


# This stuff was developed for the multi-headed attention.

def with_pad_and_mask(layer):
    def create_model_input_forward(Xs, drop=0.):
        nX = model.ops.asarray([x.shape[0] for x in Xs], dtype='i')
        nL = nX.max()
        X, unpad_X = pad_sequences(model.ops, Xs, pad_to=nL)
        X_mask = _get_mask(X, nX)
        Y, bp_Y = layer.begin_update((X.astype("float32"), X_mask, None), drop=drop)
        def create_model_input_backward(dYs, sgd=None):
            dY, _ = pad_sequences(model.ops, dYs, pad_to=nL)
            dX = bp_Y(dY, sgd=sgd)
            return unpad_X(dX)
        return unpad_X(Y), create_model_input_backward
    model = layerize(create_model_input_forward)
    return model


def pad_sequences(ops, seqs_in, pad_to=None):
    lengths = ops.asarray([len(seq) for seq in seqs_in], dtype='i')
    nB = len(seqs_in)
    if pad_to is None:
        pad_to = lengths.max()
    arr = ops.allocate((nB, int(pad_to)) + seqs_in[0].shape[1:], dtype=seqs_in[0].dtype)
    for arr_i, seq in enumerate(seqs_in):
        arr[arr_i, :seq.shape[0]] = ops.asarray(seq)
    
    def unpad(padded):
        unpadded = [None] * len(lengths)
        for i in range(padded.shape[0]):
            unpadded[i] = padded[i, :lengths[i]]
        return unpadded
    return arr, unpad


def _get_mask(X, nX):
    nB = X.shape[0]
    nL = X.shape[1]
    X_mask = Model.ops.allocate((nB, nL, nL))
    for i, length in enumerate(nX):
        X_mask[i, :, :length] = 1.0
    return X_mask



