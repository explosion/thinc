from typing import Callable, Tuple, Optional

from ..model import Model
from ..config import registry
from ..types import Array2d
from ..initializers import uniform_init


InT = Array2d
OutT = Array2d


@registry.layers("HashEmbed.v0")
def HashEmbed(
    nO: int,
    nV: int,
    *,
    seed: Optional[int] = None,
    column: int = 0,
    initializer: Callable = uniform_init,
) -> Model[InT, OutT]:
    model: Model[InT, OutT] = Model(
        "hashembed",
        forward,
        init=create_init(initializer),
        params={"E": None},
        dims={"nO": nO, "nV": nV, "nI": None},
        attrs={"seed": seed, "column": column},
    )
    if seed is None:
        model.set_attr("seed", model.id)
    model.initialize()
    return model


def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    E = model.get_param("E")
    seed = model.get_attr("seed")
    column = model.get_attr("column")
    nV = E.shape[0]
    input_shape = tuple(ids.shape)
    if ids.ndim >= 2:
        ids = model.ops.xp.ascontiguousarray(ids[:, column], dtype="uint64")
    keys = model.ops.hash(ids, seed) % nV
    output = E[keys].sum(axis=1)

    def backprop(d_output: OutT) -> InT:
        keys = model.ops.hash(ids, seed) % nV
        dE = model.ops.alloc_f2d(*E.shape)
        keys = model.ops.xp.ascontiguousarray(keys.T, dtype="i")
        for i in range(keys.shape[0]):
            model.ops.scatter_add(dE, keys[i], d_output)
        model.inc_grad("E", dE)
        dX = model.ops.alloc(input_shape, dtype="i")
        return dX

    return output, backprop


class CreateInit(object):
    """Create an init function, given a dictionary of parameter initializers."""

    def __init__(self, initializer: Callable):
        self.initializer = initializer

    def __call__(self, model: Model, X: Optional[InT] = None, Y: Optional[OutT] = None) -> Model:
        vectors = model.ops.alloc_f2d(model.get_dim("nV"), model.get_dim("nO"))
        self.initializer(vectors, inplace=True)
        model.set_param("E", vectors)
        return model
