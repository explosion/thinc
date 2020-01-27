from typing import Dict, Callable, Tuple, Optional, Any, Union, cast

from ..model import Model
from ..config import registry
from ..types import Ints2d, Floats2d, Ints1d
from ..initializers import uniform_init
from ..util import get_width, partial


InT = Union[Ints1d, Ints2d]
OutT = Floats2d


@registry.layers("Embed.v1")
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
    vectors = cast(Floats2d, model.get_param("E"))
    column: int = model.attrs["column"]
    dropout = model.attrs.get("dropout_rate")
    input_shape = tuple(ids.shape)
    if ids.ndim == 2:
        ids1d = ids[:, column]  # type: ignore
    else:
        ids1d = cast(Ints1d, ids)
    ids1d *= ids1d < nV
    output = vectors[ids1d.astype("i")]
    drop_mask = cast(Floats2d, model.ops.get_dropout_mask((vectors.shape[1],), dropout))
    output *= drop_mask

    def backprop(d_output: OutT) -> InT:
        d_output *= drop_mask
        d_vectors = model.ops.alloc2f(*vectors.shape)
        model.ops.scatter_add(d_vectors, ids1d, d_output)
        model.inc_grad("E", d_vectors)
        dX: InT = model.ops.alloc(input_shape, dtype=ids1d.dtype)
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
