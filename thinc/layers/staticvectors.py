from typing import Tuple, Callable, Optional, cast, Union, Dict, Any

from thinc.initializers import glorot_uniform_init
from thinc.util import partial

from .chain import chain
from .array_getitem import ints_getitem
from ..types import Ints1d, Floats2d, Ints2d, Floats1d, Unserializable
from ..model import Model
from ..backends import Ops
from ..config import registry
from contextvars import ContextVar


InT = Union[Ints1d, Ints2d]
OutT = Floats2d

context_vectors: ContextVar[dict] = ContextVar("context_vectors", default={})


@registry.layers("StaticVectors.v1")
def StaticVectors(
    nO: Optional[int] = None,
    *,
    vectors_name: Optional[str] = None,
    vectors: Optional[Floats2d] = None,
    key2row: Optional[Dict[int, int]] = None,
    oov_row: int=0,
    column: Optional[int] = None,
    dropout: Optional[float] = None,
    init_W: Callable = glorot_uniform_init,
) -> Model[InT, OutT]:
    """Embed integer IDs using a static vectors table and a learned linear
    mapping.

    To prevent data duplication, the vectors data is not serialized within the
    layer. This means you'll often need to set the vectors after the layer has
    been created. Standard usage is to pass in the name of the vectors, so you
    can more accurately find the layers in that require a particular
    vectors table.
    
    Arguments:

    nO (int): The output width.
    column (int): If the input data is a 2d array, use this column as the input.
    vectors_name (str or None): An identifier string for the vectors.
    vectors (Floats2d or None): The vectors data.
    key2row (dict or None): A dictionary mapping the incoming keys to rows in
        the vectors table. If None, an identity mapping is assumed.
    oov_row (int): The row to map unseen keys to, if the key2row mapping is
        used.
    """
    attrs: Dict[str, Any] = {
        "column": column,
        "vectors_name": vectors_name,
        "vectors": vectors if vectors is None else Unserializable(vectors),
        "key2row": key2row if key2row is None else Unserializable(key2row),
        "oov_row": oov_row
    }
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
    model: Model[InT, OutT], keys: Ints1d, is_train: bool
) -> Tuple[OutT, Callable]:
    if model.attrs.get("vectors") is None:
        raise AttributeError("StaticVectors has vectors unset.")
    # Assume the original 'vectors' object contains the actual data and is compatible with Floats2d
    vectors = cast(Floats2d, model.attrs["vectors"].obj)
    if model.attrs.get("key2row") is None:
        ids = keys
    else:
        ids = get_rows(model.ops, keys, model.attrs["key2row"].obj)
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


def get_rows(ops: Ops, keys: Ints1d, key2row: Dict[int, int], oov: int) -> Ints1d:
    keys = ops.to_numpy(keys)
    return ops.asarray(
        [key2row.get(key, oov) for key in ops.to_numpy(keys)],
        dtype=keys.dtype
    )
