from typing import Callable, Tuple, Optional, TypeVar

from ..model import Model
from ..types import Array
from ..initializers import uniform_init
from ..util import get_width


InputType = TypeVar("InputType", bound=Array)
OutputType = TypeVar("OutputType", bound=Array)


def Embed(
    nO: Optional[int] = None,
    nV: Optional[int] = None,
    *,
    column: int = 0,
    initializer: Callable = uniform_init,
) -> Model:
    """Map integers to vectors, using a fixed-size lookup table."""
    return Model(
        "embed",
        forward,
        init=create_init(initializer),
        dims={"nO": nO, "nV": nV},
        attrs={"column": column},
        params={"vectors": None},
    )


def forward(
    model: Model, ids: Array, is_train: bool
) -> Tuple[Array, Callable]:
    nV = model.get_dim("nV")
    vectors = model.get_param("vectors")
    column = model.get_attr("column")
    if ids.ndim == 2:
        ids = ids[:, column]
    ids[ids >= nV] = 0
    output = vectors[ids]

    def backprop(d_output: Array) -> Array:
        d_vectors = model.ops.allocate(vectors.shape)
        model.ops.scatter_add(d_vectors, ids, d_output)
        model.inc_grad("vectors", d_vectors)
        return model.ops.allocate(ids.shape, dtype=ids.dtype)

    return output, backprop


def create_init(initializer: Callable) -> Callable:
    def init(
        model: Model, X: Optional[Array] = None, Y: Optional[Array] = None
    ) -> None:
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        shape = (model.get_dim("nV"), model.get_dim("nO"))
        vectors = initializer(model.ops.allocate(shape))
        model.set_param("vectors", vectors)

    return init
