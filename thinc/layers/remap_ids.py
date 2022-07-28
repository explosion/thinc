from typing import Tuple, Callable, Sequence, cast
from typing import Dict, Hashable, Union, Optional

from ..model import Model
from ..config import registry
from ..types import Ints2d, DTypes
from ..util import is_cupy_array, is_xp_array, to_numpy

InT = Union[Sequence[Hashable], Ints2d]
OutT = Ints2d


@registry.layers("remap_ids.v1")
def remap_ids(
    mapping_table: Dict[Hashable, int] = {},
    default: int = 0,
    dtype: DTypes = "i",
    *,
    column: Optional[int] = None
) -> Model[InT, OutT]:
    """Remap string or integer inputs using a mapping table, usually as a
    preprocess before embeddings. The mapping table can be passed in on input,
    or updated after the layer has been created. The mapping table is stored in
    the "mapping_table" attribute.
    """
    return Model(
        "remap_ids",
        forward,
        attrs={
            "mapping_table": mapping_table,
            "dtype": dtype,
            "default": default,
            "column": column
        },
    )


def forward(
    model: Model[InT, OutT], inputs: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    table = model.attrs["mapping_table"]
    default = model.attrs["default"]
    dtype = model.attrs["dtype"]
    column = model.attrs["column"]
    if column is not None:
        inputs = cast(Ints2d, inputs)
        inputs = inputs[:, column]
    # elements of cupy arrays are 0-dimensional arrays
    # not the integers stored in the original mapper.
    idx = to_numpy(inputs)  # type: ignore
    values = [table.get(x, default) for x in idx]
    arr = model.ops.asarray2i(values, dtype=dtype)
    output = model.ops.reshape2i(arr, -1, 1)

    def backprop(dY: OutT) -> InT:
        if is_xp_array(inputs):
            return model.ops.xp.empty(dY.shape)  # type: ignore
        else:
            return []

    return output, backprop
