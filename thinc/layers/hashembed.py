from typing import Callable, Tuple, Optional, cast

from ..model import Model
from ..config import registry
from ..types import Ints2d, Floats2d
from ..initializers import uniform_init


InT = Ints2d
OutT = Floats2d


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
        params={"vectors": None},
        dims={"nO": nO, "nV": nV},
        attrs={"seed": seed, "column": column},
    )
    if seed is None:
        model.set_attr("seed", model.id)
    model.initialize()
    return model


def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    vectors = model.get_param("vectors")
    seed = model.get_attr("seed")
    column = model.get_attr("column")
    nV = vectors.shape[0]
    if ids.ndim >= 2:
        ids = model.ops.xp.ascontiguousarray(ids[:, column], dtype="uint64")
    keys = model.ops.hash(ids, seed) % nV
    output = vectors[keys].sum(axis=1)

    def backprop(d_output: OutT) -> InT:
        keys = model.ops.hash(ids, seed) % nV
        d_vectors = model.ops.alloc_f2d(*vectors.shape)
        keys = model.ops.xp.ascontiguousarray(keys.T, dtype="i")
        for i in range(keys.shape[0]):
            model.ops.scatter_add(d_vectors, keys[i], d_output)
        model.inc_grad("vectors", d_vectors)
        dX = model.ops.alloc_i2d(*ids.shape)
        return dX

    return output, backprop


def create_init(initializer: Callable) -> Callable:
    def init(model: Model, X: Optional[InT] = None, Y: Optional[OutT] = None) -> Model:
        vectors = model.ops.alloc_f2d(model.get_dim("nV"), model.get_dim("nO"))
        initializer(vectors, inplace=True)
        model.set_param("vectors", vectors)
        return model

    return init
