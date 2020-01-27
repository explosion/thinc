from typing import Tuple, Callable, Optional, cast

from ..types import Floats2d, Ints2d, Unserializable
from ..model import Model
from ..config import registry
from contextvars import ContextVar


InT = Ints2d
OutT = Floats2d

context_vectors: ContextVar[dict] = ContextVar("context_vectors", default={})


@registry.layers("StaticVectors.v1")
def StaticVectors(
    nO: Optional[int] = None,
    vectors: Optional[Floats2d] = None,
    *,
    column: int = 0,
    dropout: Optional[float] = None
) -> Model[InT, OutT]:
    attrs = {"column": column, "vectors": Unserializable(vectors)}
    if dropout is not None:
        attrs["dropout_rate"] = dropout
    return Model(
        "static_vectors",
        forward,
        init=init,
        params={"W": None},
        attrs=attrs,
        dims={"nM": None, "nV": None, "nO": nO},
    )


def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    dropout = model.attrs.get("dropout_rate")
    column = model.attrs["column"]
    W = cast(Floats2d, model.get_param("W"))
    vector_table = model.attrs["vectors"].data
    if ids.ndim >= 2:
        ids = model.ops.as_contig(ids[:, column])
    vectors = vector_table[ids * (ids < vector_table.shape[0])]
    vectors = model.ops.as_contig(vectors)
    drop_mask = model.ops.get_dropout_mask((vectors.shape[1],), dropout)
    vectors *= drop_mask
    assert vectors.shape[0] == ids.shape[0]

    def backprop(d_output: OutT) -> InT:
        model.inc_grad("W", model.ops.gemm(d_output, vectors, trans1=True))
        return model.ops.alloc(ids.shape, dtype=ids.dtype)

    output = model.ops.gemm(vectors, W, trans2=True)
    return output, backprop


def init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    vector_table = model.attrs["vectors"].data
    model.set_dim("nV", vector_table.shape[0])
    model.set_dim("nM", vector_table.shape[1])
    W = model.ops.alloc2f(model.get_dim("nO"), model.get_dim("nM"))
    model.set_param("W", W)
    return model
