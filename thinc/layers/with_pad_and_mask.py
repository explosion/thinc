from typing import List
from ..model import Model
from ..types import Array


def with_pad_and_mask(layer):
    """Wrap a layer so that list inputs are transformed into padded batches.
    The inputs are provided as (data, mask) tuples.
    """
    return Model(f"with_pad_and_mask-{layer.name}", forward, init=init, layers=[layer])


def init(model, X=None, Y=None):
    if X is not None:
        nX = model.ops.asarray([x.shape[0] for x in X], dtype="i")
        nL = nX.max()
        X, unpad_X = model.ops.pad_sequences(X, pad_to=nL)
    if Y is not None:
        nY = model.ops.asarray([y.shape[0] for y in Y], dtype="i")
        nL = nY.max()
        Y, unpad_Y = model.ops.pad_sequences(Y, pad_to=nL)
    model.layers[0].initialize(X=X, Y=Y)


def forward(model, Xs: List[Array], is_train):
    nX = model.ops.asarray([x.shape[0] for x in Xs], dtype="i")
    nL = nX.max()
    X, unpad_X = model.ops.pad_sequences(Xs, pad_to=nL)
    X_mask = _get_mask(model.ops, X, nX)
    Y, bp_Y = model.layers[0]((X.astype("float32"), X_mask), is_train)

    def backprop(dYs):
        dY, _ = model.ops.pad_sequences(dYs, pad_to=nL)
        return unpad_X(bp_Y(dY))

    return unpad_X(Y), backprop


def _get_mask(ops, X, nX):
    nB = X.shape[0]
    nL = X.shape[1]
    X_mask = ops.allocate((nB, nL, nL))
    for i, length in enumerate(nX):
        X_mask[i, :, :length] = 1.0
    return X_mask
