from typing import Tuple, Callable, Sequence, cast
from typing import Dict, Union, Optional

from ..model import Model
from ..config import registry
from ..types import Ints1d, Ints2d, ArrayXd
from ..util import is_xp_array, to_numpy


InT = Union[Sequence[str], Sequence[int], Ints1d, Ints2d]
OutT = Ints2d
IntVecOrMat = Union[Ints1d, Ints2d]


def _check_1d_or_2d_str_or_int(arr: IntVecOrMat) -> None:
    if arr.ndim > 2:
        raise ValueError(
            "Inputs array can be only one or two dimensional."
        )
    if arr.dtype.kind not in {"U", "i"}:  # type: ignore
        raise ValueError(
            "Input array has to contain strings or integers"
        )


def _check_column(arr: Ints2d, column: int) -> None:
    if column is None:
        raise ValueError(
            "For two dimensional input 'column' attribute has to be set"
        )
    if column > arr.shape[1] - 1:
        raise ValueError(
            f"Column index {column} is greater than input "
            f"dimension {arr.shape}."
        )


@registry.layers("remap_ids.v1")
def remap_ids(
    mapping_table: Optional[
        Union[Dict[int, int], Dict[str, int]]
    ] = None,
    default: int = 0,
    *,
    column: Optional[int] = None
) -> Model[InT, OutT]:
    """Remap string or integer inputs using a mapping table,
    usually as a preprocessing step before embeddings.
    The mapping table can be passed in on input,
    or updated after the layer has been created.
    The mapping table is stored in the "mapping_table" attribute.
    Two dimensional arrays can be provided as input in which case
    the 'column' chooses which column to process. This is useful
    to work together with FeatureExtractor in spaCy.
    """
    return Model(
        "remap_ids",
        forward,
        attrs={
            "mapping_table": mapping_table,
            "default": default,
            "column": column
        },
    )


def forward(
    model: Model[InT, OutT], inputs: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    table = model.attrs["mapping_table"]
    if table is None:
        raise ValueError(
            "'mapping table' not set"
        )
    default = model.attrs["default"]
    column = model.attrs["column"]
    if is_xp_array(inputs):
        xp_inputs = cast(IntVecOrMat, inputs)
        xp_input = True
        _check_1d_or_2d_str_or_int(xp_inputs)
        if xp_inputs.ndim == 2:
            _check_column(xp_inputs, column)
            idx = to_numpy(xp_inputs[:, column])
        else:
            idx = to_numpy(xp_inputs)
    else:
        xp_input = False
        idx = inputs
    values = [table.get(x, default) for x in idx]
    arr = model.ops.asarray2i(values, dtype='i')
    output = model.ops.reshape2i(arr, -1, 1)

    def backprop(dY: OutT) -> InT:
        if xp_input:
            return model.ops.xp.empty(dY.shape)  # type: ignore
        else:
            return []

    return output, backprop
