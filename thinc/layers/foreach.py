from .base import Model


def ForEach(layer):
    """Map a layer across list items"""
    return Model(
        f"foreach-{layer.name}",
        forward,
        init=init,
        dims={"nO": layer.get_dim("nO"), "nI": layer.get_dim("nI")},
        params={},
        attrs={},
    )


def init(model, X=None, Y=None):
    Xflat = [X[0]] if X else None
    Yflat = [Y[0]] if Y else None
    model._layers[0].initialize(X=Xflat, Y=Yflat)
    model.set_dim("nO", model._layers[0].get_dim("nO"))
    model.set_dim("nI", model._layers[0].get_dim("nI"))


def forward(model, docs, is_train):
    layer = model._layers[0]
    sents = []
    lengths = []
    for doc in docs:
        doc_sents = [sent for sent in doc if len(sent)]
        sents.extend(doc_sents)
        lengths.append(len(doc_sents))
    assert len(sents)
    flat, bp_flat = layer(sents, is_train)
    output = layer.ops.unflatten(flat, lengths)

    def foreach_bwd(d_output):
        d_flat = layer.ops.flatten(d_output)
        d_sents = bp_flat(d_flat)
        return layer.ops.unflatten(d_sents, lengths)

    return output, foreach_bwd
