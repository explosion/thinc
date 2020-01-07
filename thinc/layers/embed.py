from typing import Callable, Tuple, Optional, cast

from ..model import Model
from ..config import registry
from ..types import Ints2d, Floats2d
from ..initializers import uniform_init
from ..util import get_width


InT = Ints2d
OutT = Floats2d


@registry.layers("Embed.v0")
def Embed(
    nO: Optional[int] = None,
    nV: Optional[int] = None,
    *,
    column: int = 0,
    initializer: Callable = uniform_init,
) -> Model[InT, OutT]:
    """Map integers to vectors, using a fixed-size lookup table."""
    return Model(
        "embed",
        forward,
        init=create_init(initializer),
        dims={"nO": nO, "nV": nV},
        attrs={"column": column},
        params={"vectors": None},
    )


def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    nV = model.get_dim("nV")
    vectors = model.get_param("vectors")
    column = model.get_attr("column")
    if ids.ndim == 2:
        ids = ids[:, column]
    ids[ids >= nV] = 0
    output = vectors[ids]

    def backprop(d_output: OutT) -> InT:
        d_vectors: Floats2d = model.ops.alloc(vectors.shape)
        model.ops.scatter_add(d_vectors, ids, d_output)
        model.inc_grad("vectors", d_vectors)
        dX: Ints2d = model.ops.alloc(ids.shape, dtype=ids.dtype)
        return dX

    return output, backprop


def create_init(initializer: Callable) -> Callable:
    def init(
        model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
    ) -> None:
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        shape = (model.get_dim("nV"), model.get_dim("nO"))
        vectors = initializer(model.ops.alloc(shape))
        model.set_param("vectors", vectors)

    return init
