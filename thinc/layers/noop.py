from typing import Tuple, Callable, List, Any

from ..model import Model


def noop(*layers: List[Model]) -> Model:
    """Transform a sequences of layers into a null operation."""
    return Model(forward, layers=layers)


def forward(model: Model, X: Any, is_train: bool) -> Tuple[Any, Callable]:
    return X, lambda dY: dY
