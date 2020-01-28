from typing import Tuple, Callable, Optional, cast, Union, Dict, Any

from .chain import chain
from .array_getitem import ints_getitem
from ..types import Ints1d, Floats2d, Ints2d, Floats1d, Unserializable
from ..model import Model
from ..config import registry
from contextvars import ContextVar


InT = Union[Ints1d, Ints2d]
OutT = Floats2d

context_vectors: ContextVar[dict] = ContextVar("context_vectors", default={})


@registry.layers("StaticVectors.v1")
def StaticVectors(
    nO: Optional[int] = None,
    vectors: Optional[Floats2d] = None,
    *,
    column: Optional[int] = None,
    dropout: Optional[float] = None
) -> Model[InT, OutT]:
    attrs: Dict[str, Any] = {"column": column, "vectors": Unserializable(vectors)}
    if dropout is not None:
        attrs["dropout_rate"] = dropout
    model = Model(  # type: ignore
        "static_vectors",
        forward,
        init=init,
        params={"W": None},
        attrs=attrs,
        dims={"nM": None, "nV": None, "nO": nO},
    )
    if column is not None:
        # This is equivalent to array[:, column]. What you're actually doing
        # there is passing in a tuple: array[(:, column)], except in the context
        # of array indexing, the ":" creates an object slice(0, None).
        # So array[:, column] is array.__getitem__(slice(0), column).
        model = chain(ints_getitem((slice(0, None), column)), model)
    model.attrs["column"] = column
    return cast(Model[InT, OutT], model)


def forward(
    model: Model[InT, OutT], ids: Ints1d, is_train: bool
) -> Tuple[OutT, Callable]:
    vectors = cast(Floats2d, model.attrs["vectors"].data)
    nO = vectors.shape[1]
    nN = ids.shape[0]
    dropout: Optional[float] = model.attrs.get("dropout_rate")
    output = vectors[ids]
    drop_mask = cast(Floats1d, model.ops.get_dropout_mask((nO,), dropout))
    output *= drop_mask
    W = cast(Floats2d, model.get_param("W"))
    vec = vectors[ids * (ids < vectors.shape[0])]
    vec = model.ops.as_contig(vec)
    assert vec.shape[0] == ids.shape[0]

    def backprop(d_output: OutT) -> Ints1d:
        model.inc_grad("W", model.ops.gemm(d_output, vectors, trans1=True))
        dX = model.ops.alloc1i(nN)
        return dX

    output = model.ops.gemm(vectors, W, trans2=True)
    return output, backprop


def init(
    model: Model[InT, OutT], X: Optional[Ints1d] = None, Y: Optional[OutT] = None
) -> Model[InT, OutT]:
    vector_table = model.attrs["vectors"].data
    if vector_table is None:
        raise ValueError("Can't initialize: vectors attribute unset")
    model.set_dim("nV", vector_table.shape[0] + 1)
    model.set_dim("nM", vector_table.shape[1])
    W = model.ops.alloc2f(model.get_dim("nO"), model.get_dim("nM"))
    model.set_param("W", W)
    return model
