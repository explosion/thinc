from typing import List, Union, Callable, Tuple, TypeVar
from ..types import Array, DocType
from ..model import Model


# TODO: fix and make more specific
InT = TypeVar("InT", bound=List[DocType])
OutputValue = TypeVar("OutputValue", bound=Array)
OutT = List[OutputValue]


def FeatureExtractor(columns: List[Union[int, str]]) -> Model[InT, OutT]:
    return Model("extract_features", forward, attrs={"columns": columns})


def forward(
    model: Model[InT, OutT], docs: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    columns = model.get_attr("columns")
    features: OutT = []
    for doc in docs:
        if hasattr(doc, "to_array"):
            attrs = doc.to_array(columns)
        else:
            attrs = doc.doc.to_array(columns)[doc.start : doc.end]
        features.append(model.ops.asarray(attrs, dtype="uint64"))

    backprop: Callable[[OutT], List] = lambda d_features: []
    return features, backprop
