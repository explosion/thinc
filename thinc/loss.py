from typing import Tuple, List, cast, TypeVar, Generic, Any

from .types import Array2d, Array
from .util import get_array_module, to_categorical
from .config import registry


LossT = TypeVar("LossT")
GradT = TypeVar("GradT")
GuessT = TypeVar("GuessT")
TruthT = TypeVar("TruthT")


class Loss(Generic[GuessT, TruthT, GradT, LossT]):  # pragma: no cover
    """Base class for classes computing the loss / gradient. The class can
    be initialized with settings if needed. It provides get_loss and
    get_grad as separate methods to allow calculating them separately. It
    also provides a __call__ method that returns a tuple of both.
    """

    def __init__(self, **kwargs: Any) -> None:
        ...

    def __call__(self, guesses: GuessT, truths: TruthT) -> Tuple[GradT, LossT]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_grad(self, guesses: GuessT, truths: TruthT) -> GradT:
        ...

    def get_loss(self, guesses: GuessT, truths: TruthT) -> LossT:
        ...


class CategoricalCrossentropy(Loss):
    def __call__(self, guesses: Array2d, truths: Array) -> Tuple[Array2d, float]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_grad(self, guesses: Array2d, truths: Array) -> Array2d:
        if truths.ndim != guesses.ndim:
            target = to_categorical(truths, n_classes=guesses.shape[-1])
        else:  # pragma: no cover
            target = cast(Array2d, truths)
        if guesses.shape != target.shape:  # pragma: no cover
            err = f"Cannot calculate loss: mismatched shapes. {guesses.shape} vs {target.shape}"
            raise ValueError(err)
        difference = guesses - target
        return difference / guesses.shape[0]

    def get_loss(self, guesses: Array2d, truths: Array) -> float:
        raise NotImplementedError


@registry.losses("CategoricalCrossentropy.v0")
def configure_CategoricalCrossentropy() -> CategoricalCrossentropy:
    return CategoricalCrossentropy()


class SequenceCategoricalCrossentropy(Loss):
    def __init__(self):
        self.cc = CategoricalCrossentropy()

    def __call__(
        self, guesses: List[Array2d], truths: List[Array]
    ) -> Tuple[List[Array2d], List[float]]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_grad(self, guesses: List[Array2d], truths: List[Array]) -> List[Array2d]:
        if not guesses:
            return []
        if len(guesses) != len(truths):  # pragma: no cover
            raise ValueError("Scores and labels must be same length")
        d_scores = []
        for yh, y in zip(guesses, truths):
            d_scores.append(self.cc.get_grad(yh, y))
        return d_scores

    def get_loss(self, guesses: List[Array2d], truths: List[Array]) -> List[float]:
        raise NotImplementedError


@registry.losses("SequenceCategoricalCrossentropy.v0")
def configure_SequenceCategoricalCrossentropy() -> SequenceCategoricalCrossentropy:
    return SequenceCategoricalCrossentropy()


class L1Distance(Loss):
    def __init__(self, *, margin: float = 0.2):
        self.margin = margin

    def __call__(
        self, guesses: Tuple[Array2d, Array2d], truths: Array2d
    ) -> Tuple[Tuple[Array2d, Array2d], float]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_grad(
        self, guesses: Tuple[Array2d, Array2d], truths: Array2d
    ) -> Tuple[Array2d, Array2d]:
        vec1, vec2 = guesses
        xp = get_array_module(vec1)
        dist = xp.abs(vec1 - vec2).sum(axis=1)
        loss = (dist > self.margin) - truths
        return (vec1 - vec2) * loss, (vec2 - vec1) * loss

    def get_loss(self, guesses: Tuple[Array2d, Array2d], truths: Array2d) -> float:
        raise NotImplementedError


@registry.losses("L1Distance.v0")
def configure_L1Distance(*, margin: float = 0.2) -> L1Distance:
    return L1Distance(margin=margin)


class CosineDistance(Loss):
    def __init__(self, *, ignore_zeros: bool = False):
        self.ignore_zeros = bool

    def __call__(self, guesses: Array2d, truths: Array2d) -> Tuple[Array2d, float]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_grad(self, guesses: Array2d, truths: Array2d) -> Array2d:
        xp = get_array_module(guesses)
        # Find the zero vectors
        if self.ignore_zeros:
            zero_indices = xp.abs(truths).sum(axis=1) == 0
        # Add a small constant to avoid 0 vectors
        yh = guesses + 1e-8
        y = truths + 1e-8
        # https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosinesimilarity
        norm_yh = xp.linalg.norm(yh, axis=1, keepdims=True)
        norm_y = xp.linalg.norm(y, axis=1, keepdims=True)
        mul_norms = norm_yh * norm_y
        cosine = (yh * y).sum(axis=1, keepdims=True) / mul_norms
        d_yh = (y / mul_norms) - (cosine * (yh / norm_yh ** 2))
        losses = xp.abs(cosine - 1)
        if self.ignore_zeros:
            # If the target was a zero vector, don't count it in the loss.
            d_yh[zero_indices] = 0
            losses[zero_indices] = 0
        return -d_yh

    def get_loss(self, guesses: Array2d, truths: Array2d) -> float:
        raise NotImplementedError


@registry.losses("CosineDistance.v0")
def configure_CosineDistance(*, ignore_zeros: bool = False) -> CosineDistance:
    return CosineDistance(ignore_zeros=ignore_zeros)


__all__ = ["CategoricalCrossentropy", "L1Distance", "CosineDistance"]
