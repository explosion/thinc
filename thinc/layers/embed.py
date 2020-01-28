from typing import Dict, Callable, Tuple, Optional, Any, Union, cast

from .chain import chain
from .array_getitem import ints_getitem
from ..model import Model
from ..config import registry
from ..types import Ints1d, Ints2d, Floats1d, Floats2d, Ints1d
from ..initializers import uniform_init
from ..util import get_width, partial


InT = Ints1d
OutT = Floats2d


@registry.layers("Embed.v1")
def Embed(
    nO: Optional[int] = None,
    nV: Optional[int] = None,
    *,
    column: Optional[int] = None,
    initializer: Callable = uniform_init,
    dropout: Optional[float] = None
) -> Model[InT, OutT]:
    """Map integers to vectors, using a fixed-size lookup table."""
    attrs: Dict[str, Union[None, int, float]] = {}
    if dropout is not None:
        attrs["dropout_rate"] = dropout
    model = Model( # type: ignore
        "embed",
        forward,
        init=partial(init, initializer),
        attrs=attrs,
        dims={"nO": nO, "nV": nV},
        params={"E": None},
    )
    model = model if column is None else chain(ints_getitem(column), model)
    model.attrs["column"] = column
    return cast(Model[InT, OutT], model)



def forward(model: Model[InT, OutT], ids: InT, is_train: bool) -> Tuple[OutT, Callable]:
    vectors = cast(Floats2d, model.get_param("E"))
    nO = vectors.shape[1]
    nN = ids.shape[0]
    dropout: Optional[float] = model.attrs.get("dropout_rate")
    output = vectors[ids]
    drop_mask = cast(Floats1d, model.ops.get_dropout_mask((nO,), dropout))
    output *= drop_mask

    def backprop(d_output: OutT) -> InT:
        d_output *= drop_mask
        d_vectors = model.ops.alloc2f(*vectors.shape)
        model.ops.scatter_add(d_vectors, ids, d_output)
        model.inc_grad("E", d_vectors)
        dX = model.ops.alloc1i(nN)
        return dX

    return output, backprop


def init(
    initializer: Callable,
    model: Model[InT, OutT],
    X: Optional[InT] = None,
    Y: Optional[OutT] = None,
) -> Model[InT, OutT]:
    if Y is not None:
        model.set_dim("nO", get_width(Y))
    shape = (model.get_dim("nV")+1, model.get_dim("nO"))
    model.set_param("E", initializer(model.ops, shape))
    return model
