from .neural._classes.function_layer import FunctionLayer
from .neural._classes.function_layer import wrap
from .model import Model
from .util import is_ragged


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
