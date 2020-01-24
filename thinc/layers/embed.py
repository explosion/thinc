from typing import Dict, Callable, Tuple, Optional, Any

from ..model import Model
from ..config import registry
from ..types import Array2d
from ..initializers import uniform_init
from ..util import get_width, partial


InT = Array2d
OutT = Array2d


@registry.layers("Embed.v0")
def Embed(
    nO: Optional[int] = None,
    nV: Optional[int] = None,
    *,
    column: int = 0,
    initializer: Callable = uniform_init,
    dropout: Optional[float] = None
) -> Model[InT, OutT]:
    """Map integers to vectors, using a fixed-size lookup table."""
    attrs: Dict[str, Any] = {"column": column}
    if dropout is not None:
        attrs["dropout_rate"] = dropout
    return Model(
        "embed",
        forward,
        init=partial(init, initializer),
        attrs=attrs,
        dims={"nO": nO, "nV": nV},
        params={"E": None},
    )


def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    nV = model.get_dim("nV")
    vectors = model.get_param("E")
    column = model.get_attr("column")
    if model.has_attr("dropout_rate"):
        dropout = model.get_attr("dropout_rate")
    else:
        dropout = None
    input_shape = tuple(ids.shape)
    if ids.ndim == 2:
        ids = ids[:, column]
    ids *= ids < nV
    output = vectors[ids.astype("i")]
    drop_mask = model.ops.get_dropout_mask((vectors.shape[1],), dropout)
    output *= drop_mask

    def backprop(d_output: OutT) -> InT:
        d_output *= drop_mask
        d_vectors = model.ops.alloc_f2d(*vectors.shape)
        model.ops.scatter_add(d_vectors, ids, d_output)
        model.inc_grad("E", d_vectors)
        dX: OutT = model.ops.alloc(input_shape, dtype=ids.dtype)
        return dX

    return output, backprop


def init(
    initializer: Callable,
    model: Model[InT, OutT],
    X: Optional[InT] = None,
    Y: Optional[OutT] = None,
) -> Model[InT, OutT]:
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    shape = (model.get_dim("nV"), model.get_dim("nO"))
    model.set_param("E", initializer(model.ops, shape))
    return model
