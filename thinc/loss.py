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


def cosine_distance(yh, y, ignore_zeros=False):
    xp = get_array_module(yh)
    # Find the zero vectors
    if ignore_zeros:
        zero_indices = xp.abs(y).sum(axis=1) == 0
    # Add a small constant to avoid 0 vectors
    yh = yh + 1e-8
    y = y + 1e-8
    # https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosinesimilarity
    norm_yh = xp.linalg.norm(yh, axis=1, keepdims=True)
    norm_y = xp.linalg.norm(y, axis=1, keepdims=True)
    mul_norms = norm_yh * norm_y
    cosine = (yh * y).sum(axis=1, keepdims=True) / mul_norms
    d_yh = (y / mul_norms) - (cosine * (yh / norm_yh ** 2))
    losses = xp.abs(cosine - 1)
    if ignore_zeros:
        # If the target was a zero vector, don't count it in the loss.
        d_yh[zero_indices] = 0
        losses[zero_indices] = 0
    loss = losses.sum()
    return loss, -d_yh


__all__ = ["categorical_crossentropy", "L1_distance", "cosine_distance"]
