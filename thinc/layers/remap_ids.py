from typing import Tuple, Callable, Sequence, Dict, Any

from ..model import Model
from ..config import registry
from ..types import Array2d, DTypes


InT = Sequence[Any]
OutT = Array2d


@registry.layers("remap_ids.v0")
def remap_ids(
    mapping_table: Dict[Any, int] = {}, default: int = 0, dtype: DTypes = "i"
) -> Model[InT, OutT]:
    """Remap string or integer inputs using a mapping table, usually as a
    preprocess before embeddings. The mapping table can be passed in on input,
    or updated after the layer has been created. The mapping table is stored in
    the "mapping_table" attribute.
    """
    return Model(
        "remap_ids",
        forward,
        attrs={"mapping_table": mapping_table, "dtype": dtype, "default": default},
    )


def forward(
    model: Model[InT, OutT], inputs: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    table = model.get_attr("mapping_table")
    default = model.get_attr("default")
    dtype = model.get_attr("dtype")
    values = [table.get(x, default) for x in inputs]
    output = model.ops.asarray(values, dtype=dtype).reshape(-1, 1)

    def backprop(dY: OutT) -> InT:
        return []

    return output, backprop
