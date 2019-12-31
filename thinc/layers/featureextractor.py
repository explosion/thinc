from typing import List, Union, Callable, Tuple, TypeVar
from ..types import Array, DocType
from ..model import Model


InputType = TypeVar("InputType", bound=List[DocType])
OutputValue = TypeVar("OutputValue", bound=Array)
OutputType = List[OutputValue]


def FeatureExtractor(columns: List[Union[int, str]]) -> Model:
    return Model("extract_features", forward, attrs={"columns": columns})


def forward(
    model: Model, docs: InputType, is_train: bool
) -> Tuple[OutputType, Callable]:
    columns = model.get_attr("columns")
    features: OutputType = []
    for doc in docs:
        if hasattr(doc, "to_array"):
            attrs = doc.to_array(columns)
        else:
            attrs = doc.doc.to_array(columns)[doc.start : doc.end]
        features.append(model.ops.asarray(attrs, dtype="uint64"))

    backprop: Callable[[OutputType], List] = lambda d_features: []
    return features, backprop
