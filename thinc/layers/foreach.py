from typing import Tuple, Callable, Optional, Sequence

from ..model import Model
from ..types import Array


# TODO: fix and make more specific?
InT = Sequence[Array]
OutT = Sequence[Array]


def foreach(layer: Model) -> Model[InT, OutT]:
    """Map a layer across list items."""
    return Model(
        f"foreach-{layer.name}",
        forward,
        init=init,
        dims={"nO": layer.get_dim("nO"), "nI": layer.get_dim("nI")},
    )


def forward(
    model: Model[InT, OutT], docs: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    layer = model.layers[0]
    sents = []
    for doc in docs:
        sents.extend([sent for sent in doc if len(sent)])
    assert len(sents)
    lengths = model.ops.asarray([len(s) for s in sents], dtype="i")
    flat, bp_flat = layer(sents, is_train)
    output = layer.ops.unflatten(flat, lengths)

    def backprop(d_output: OutT) -> InT:
        d_flat = layer.ops.flatten(d_output)
        d_sents = bp_flat(d_flat)
        return layer.ops.unflatten(d_sents, lengths)

    return output, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> None:
    Xflat = [X[0]] if X else None
    Yflat = [Y[0]] if Y else None
    layer = model.layers[0]
    layer.initialize(X=Xflat, Y=Yflat)
    model.set_dim("nO", layer.get_dim("nO"))
    model.set_dim("nI", layer.get_dim("nI"))
