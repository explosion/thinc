from typing import Tuple

from .types import Array
from .util import get_array_module


def categorical_crossentropy(scores: Array, labels: Array) -> Tuple[Array, float]:
    xp = get_array_module(scores)
    target = xp.zeros(scores.shape, dtype="float32")
    loss = 0.0
    for i in range(len(labels)):
        target[i, int(labels[i])] = 1.0
        loss += (1.0 - scores[i, int(labels[i])]) ** 2
    return scores - target, loss


def L1_distance(
    vec1: Array, vec2: Array, labels: Array, margin: float = 0.2
) -> Tuple[Array, Array, float]:
    xp = get_array_module(vec1)
    dist = xp.abs(vec1 - vec2).sum(axis=1)
    loss = (dist > margin) - labels
    return (vec1 - vec2) * loss, (vec2 - vec1) * loss, loss
