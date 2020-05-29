from typing import Tuple, Callable, Optional, cast, Union, Dict, Any

from thinc.initializers import glorot_uniform_init
from thinc.util import partial

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
    dropout: Optional[float] = None,
    init_W: Callable = glorot_uniform_init,
) -> Model[InT, OutT]:
    attrs: Dict[str, Any] = {"column": column, "vectors": Unserializable(vectors)}
    if dropout is not None:
        attrs["dropout_rate"] = dropout
    model = Model(  # type: ignore
        "static_vectors",
        forward,
        init=partial(init, init_W),
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
    # Assume the original 'vectors' object contains the actual data and is compatible with Floats2d
    vectors = cast(Floats2d, model.attrs["vectors"].obj)
    W = cast(Floats2d, model.get_param("W"))
    nN = ids.shape[0]
    vectors = vectors[ids * (ids < vectors.shape[0])]
    vectors = model.ops.as_contig(vectors)
    assert vectors.shape[0] == ids.shape[0]

    output = model.ops.gemm(vectors, W, trans2=True)
    dropout: Optional[float] = model.attrs.get("dropout_rate")
    drop_mask = cast(Floats1d, model.ops.get_dropout_mask((output.shape[1],), dropout))

    def backprop(d_output: OutT) -> Ints1d:
        d_output *= drop_mask
        model.inc_grad("W", model.ops.gemm(d_output, vectors, trans1=True))
        dX = model.ops.alloc1i(nN)
        return dX

    output *= drop_mask
    return output, backprop


def init(
    init_W: Callable,
    model: Model[InT, OutT],
    X: Optional[InT] = None,
    Y: Optional[OutT] = None,
) -> Model[InT, OutT]:
    # Assume the original 'vectors' object contains the actual data
    vectors = model.attrs["vectors"].obj
    if vectors is None:
        raise ValueError("Can't initialize: vectors attribute unset")
    model.set_dim("nV", vectors.shape[0] + 1)
    model.set_dim("nM", vectors.shape[1])
    model.set_param("W", init_W(model.ops, (model.get_dim("nO"), model.get_dim("nM"))))
    return model
