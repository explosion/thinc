from typing import Tuple, Callable, Optional, List, Sequence, TypeVar

from ..model import Model
from ..types import Array


InputValue = TypeVar("InputValue", bound=Sequence)
InputType = List[InputValue]
OutputValue = TypeVar("OutputValue", bound=Sequence)
OutputType = List[OutputValue]


def foreach(layer: Model) -> Model:
    """Map a layer across list items."""
    return Model(
        f"foreach-{layer.name}",
        forward,
        init=init,
        dims={"nO": layer.get_dim("nO"), "nI": layer.get_dim("nI")},
    )


def forward(
    model: Model, docs: Sequence[Array], is_train: bool
) -> Tuple[Sequence[Array], Callable]:
    layer = model.layers[0]
    sents = []
    for doc in docs:
        sents.extend([sent for sent in doc if len(sent)])
    assert len(sents)
    lengths = model.ops.asarray([len(s) for s in sents], dtype="i")
    flat, bp_flat = layer(sents, is_train)
    output = layer.ops.unflatten(flat, lengths)

    def backprop(d_output: Sequence[Array]) -> Sequence[Array]:
        d_flat = layer.ops.flatten(d_output)
        d_sents = bp_flat(d_flat)
        return layer.ops.unflatten(d_sents, lengths)

    return output, backprop


def init(
    model: Model, X: Optional[InputType] = None, Y: Optional[OutputType] = None
) -> None:
    Xflat = [X[0]] if X else None
    Yflat = [Y[0]] if Y else None
    layer = model.layers[0]
    layer.initialize(X=Xflat, Y=Yflat)
    model.set_dim("nO", layer.get_dim("nO"))
    model.set_dim("nI", layer.get_dim("nI"))
