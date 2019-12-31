from typing import List, Union, Array, Callable, Tuple
from ..model import Model


def FeatureExtractor(columns: List[Union[int, str]]) -> Model:
    return Model("extract_features", forward, attrs={"columns": columns})


def forward(model: Model, docs, is_train: bool) -> Tuple[Array, Callable]:
    columns = model.get_attr("columns")
    features = []
    for doc in docs:
        if hasattr(doc, "to_array"):
            attrs = doc.to_array(columns)
        else:
            attrs = doc.doc.to_array(columns)[doc.start : doc.end]
        features.append(model.ops.asarray(attrs, dtype="uint64"))
    return features, lambda d_features: d_features
