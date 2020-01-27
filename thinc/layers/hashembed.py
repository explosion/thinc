from typing import Callable, Dict, Tuple, Optional, Any, Union, cast

from ..model import Model
from ..config import registry
from ..types import Floats2d, Ints2d, Ints1d
from ..initializers import uniform_init
from ..util import partial


InT = Union[Ints2d, Ints1d]
OutT = Floats2d


@registry.layers("HashEmbed.v1")
def HashEmbed(
    nO: int,
    nV: int,
    *,
    seed: Optional[int] = None,
    column: int = 0,
    initializer: Callable = uniform_init,
    dropout: Optional[float] = None
) -> Model[InT, OutT]:
    attrs: Dict[str, Any] = {"column": column, "seed": seed}
    if dropout is not None:
        attrs["dropout_rate"] = dropout
    model: Model[InT, OutT] = Model(
        "hashembed",
        forward,
        init=partial(init, initializer),
        params={"E": None},
        dims={"nO": nO, "nV": nV, "nI": None},
        attrs=attrs,
    )
    if seed is None:
        model.attrs["seed"] = model.id
    return model


def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    dropout = model.attrs.get("dropout_rate")
    E = cast(Floats2d, model.get_param("E"))
    seed: int = model.attrs["seed"]
    column: int = model.attrs["column"]
    nV = E.shape[0]
    input_shape = tuple(ids.shape)
    if ids.ndim >= 2:
        ids1d = model.ops.as_contig(ids[:, column], dtype="uint64")  # type: ignore
    else:
        ids1d = cast(Ints1d, ids)
    keys = model.ops.hash(ids1d, seed) % nV
    vectors = E[keys].sum(axis=1)
    drop_mask = cast(Floats2d, model.ops.get_dropout_mask((vectors.shape[1],), dropout))
    vectors *= drop_mask

    def backprop(d_vectors: OutT) -> InT:
        d_vectors *= drop_mask
        keys = model.ops.hash(ids1d, seed) % nV
        dE = model.ops.alloc2f(*E.shape)
        keys = model.ops.as_contig(keys.T, dtype="i")
        for i in range(keys.shape[0]):
            model.ops.scatter_add(dE, keys[i], d_vectors)
        model.inc_grad("E", dE)
        dX = cast(InT, model.ops.alloc_i(input_shape))
        return dX

    return vectors, backprop


def init(
    initializer: Callable,
    model: Model[InT, OutT],
    X: Optional[InT] = None,
    Y: Optional[OutT] = None,
) -> Model[InT, OutT]:
    E = initializer(model.ops, (model.get_dim("nV"), model.get_dim("nO")))
    model.set_param("E", E)
    return model
