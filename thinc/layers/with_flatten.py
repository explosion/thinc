from .base import Model


def forward(model, seqs_in, is_train):
    layer = model._layers[0]
    pad = model.get_attr("pad")

    lengths = layer.ops.asarray([len(seq) for seq in seqs_in])
    X, bp_layer = layer.begin_update(layer.ops.flatten(seqs_in, pad=pad))

    def backprop_with_flatten(d_seqs_out):
        d_X = bp_layer(layer.ops.flatten(d_seqs_out, pad=pad))
        return layer.ops.unflatten(d_X, lengths, pad=pad)

    return layer.ops.unflatten(X, lengths, pad=pad), backprop_with_flatten


def init(model, X=None, Y=None):
    layer = model._layers[0]
    pad = model.get_attr("pad")
    if X is not None:
        Xflat = layer.ops.flatten(X, pad=pad)
    if Y is not None:
        Yflat = layer.ops.flatten(Y, pad=pad)
    layer.initialize(X=Xflat, Y=Yflat)
    model.set_dim("nI", layer.get_dim("nI"))
    model.set_dim("nO", layer.get_dim("nO"))


def with_flatten(layer, pad: int = 0):
    return Model(
        f"with_flatten-{layer.name}",
        forward,
        init=init,
        layers=[layer],
        attrs={"pad": pad},
        dims={"nO": layer.get_dim("nO"), "nI": layer.get_dim("nI")},
    )
