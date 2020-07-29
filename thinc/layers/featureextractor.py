from typing import List, Union, Callable, Tuple

from ..types import Ints2d, Doc
from ..model import Model
from ..config import registry


InT = List[Doc]
OutT = List[Ints2d]


@registry.layers("FeatureExtractor.v1")
def FeatureExtractor(columns: List[Union[int, str]]) -> Model[InT, OutT]:
    return Model("extract_features", forward, attrs={"columns": columns})


def forward(model: Model[InT, OutT], docs, is_train: bool) -> Tuple[OutT, Callable]:
    columns = model.attrs["columns"]
    features: OutT = []
    for doc in docs:
        if hasattr(doc, "to_array"):
            attrs = doc.to_array(columns)
        else:
            attrs = doc.doc.to_array(columns)[doc.start : doc.end]
        if attrs.ndim == 1:
            attrs = attrs.reshape((attrs.shape[0], 1))
        features.append(model.ops.asarray2i(attrs, dtype="uint64"))

    backprop: Callable[[OutT], List] = lambda d_features: []
    return features, backprop
