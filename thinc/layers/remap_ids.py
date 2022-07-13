from typing import Tuple, Callable, Sequence
from typing import Dict, Any, Union, Optional

from ..model import Model
from ..config import registry
from ..types import Ints2d, DTypes, Array2d
from ..util import is_cupy_array, is_xp_array

InT = Union[Sequence[Any], Array2d]
OutT = Ints2d


@registry.layers("remap_ids.v1")
def remap_ids(
    mapping_table: Dict[Any, int] = {},
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
        inputs = inputs[:, column]
    # elements of cupy arrays are 0-dimensional arrays
    # not the integers stored in the original mapper.
    if is_cupy_array(inputs) and inputs.dtype.kind == 'i':
        inputs = map(int, inputs)
    values = [table.get(x, default) for x in inputs]
    arr = model.ops.asarray2i(values, dtype=dtype)
    output = model.ops.reshape2i(arr, -1, 1)

    def backprop(dY: OutT) -> InT:
        if is_xp_array(inputs):
            return model.ops.asarray([])
        else:
            return []

    return output, backprop
