from typing import Callable, Tuple, Optional

from ..model import Model
from ..config import registry
from ..types import Array2d
from ..initializers import uniform_init
from ..util import get_width


InT = Array2d
OutT = Array2d


@registry.layers("Embed.v0")
def Embed(
    nO: Optional[int] = None,
    nV: int = 1,
    *,
    column: int = 0,
    initializer: Callable = uniform_init,
) -> Model[InT, OutT]:
    """Map integers to vectors, using a fixed-size lookup table."""
    model: Model[InT, OutT] = Model(
        "embed",
        forward,
        init=create_init(initializer),
        dims={"nO": nO, "nV": nV},
        attrs={"column": column},
        params={"E": None},
    )
    if nO is not None:
        model.initialize()
    return model


def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    nV = model.get_dim("nV")
    vectors = model.get_param("E")
    column = model.get_attr("column")
    input_shape = tuple(ids.shape)
    if ids.ndim == 2:
        ids = ids[:, column]
    #ids = model.ops.xp.where(ids >= nV, 0, ids).astype("i")
    ids *= ids < nV
    output = vectors[ids.astype("i")]

    def backprop(d_output: OutT) -> InT:
        d_vectors = model.ops.alloc_f2d(*vectors.shape)
        model.ops.scatter_add(d_vectors, ids, d_output)
        model.inc_grad("E", d_vectors)
        dX = model.ops.alloc(input_shape, dtype=ids.dtype)
        return dX

    return output, backprop


def create_init(initializer: Callable) -> Callable:
    def init(
        model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
    ) -> None:
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        shape = (model.get_dim("nV"), model.get_dim("nO"))
        vectors = initializer(model.ops, shape)
        model.set_param("E", vectors)

    return init
