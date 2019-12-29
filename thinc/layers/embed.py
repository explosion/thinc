from typing import Callable, Tuple, Optional

from .base import Model
from ..types import Array
from ..initializers import uniform_init
from ..util import get_width


def Embed(
    nO: Optional[Array] = None,
    nV: Optional[Array] = None,
    column: int = 0,
    initializer: Callable = uniform_init,
) -> Model:
    return Model(
        "embed",
        forward,
        init=create_init(initializer),
        dims={"nO": nO, "nV": nV},
        attrs={"column": column},
        layers=[],
        params={"vectors": None},
    )


def create_init(initializer: Callable) -> Callable:
    def init(
        model: Model, X: Optional[Array] = None, Y: Optional[Array] = None
    ) -> None:
        if Y is not None:
            model.set_dim(get_width(Y))
        shape = (model.get_dim("nV"), model.get_dim("nO"))
        vectors = initializer(model.ops.allocate(shape))
        model.set_param("vectors", vectors)

    return init


def forward(model: Model, ids: Array, is_train: bool) -> Tuple[Array, Callable]:
    nV = model.get_dim("nV")
    vectors = model.get_param("vectors")
    column = model.get_attr("column")
    if ids.ndim == 2:
        ids = ids[:, column]
    ids[ids >= nV] = 0
    output = vectors[ids]

    def backprop_embed(d_output: Array) -> Array:
        d_vectors = model.ops.allocate(vectors.shape)
        model.ops.scatter_add(d_vectors, ids, d_output)
        model.inc_grad("vectors", d_vectors)
        return model.ops.allocate(ids.shape, dtype=ids.dtype)

    return output, backprop_embed
