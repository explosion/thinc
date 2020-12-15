from typing import Tuple, List, cast, TypeVar, Generic, Any, Union, Optional
from typing import Dict

from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
from .config import registry


LossT = TypeVar("LossT")
GradT = TypeVar("GradT")
GuessT = TypeVar("GuessT")
TruthT = TypeVar("TruthT")
IntsOrFloats = Union[Ints1d, Floats2d]
IntsOrFloatsOrStrs = Union[Ints1d, Floats2d, List[int], List[str]]


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
    names: Optional[List[str]]
    missing_value: Optional[Union[str, int]]
    _name_to_i: Dict[str, int]

    def __init__(
        self,
        *,
        normalize: bool = True,
        names: Optional[List[str]] = None,
        missing_value: Optional[Union[str, int]] = None,
    ):
        self.normalize = normalize
        self.names = names
        self.missing_value = missing_value
        if names is not None:
            self._name_to_i = {name: i for i, name in enumerate(names)}
        else:
            self._name_to_i = {}

    def convert_truths(self, truths, guesses: Floats2d) -> Tuple[Floats2d, Floats2d]:
        xp = get_array_module(guesses)
        missing = []
        missing_value = self.missing_value
        # Convert list of ints or list of strings
        if isinstance(truths, list):
            truths = list(truths)
            if len(truths) and not isinstance(truths[0], int):
                if self.names is None:
                    msg = (
                        "Cannot calculate loss from list of strings without names. "
                        "You can pass the names as a keyword argument when you "
                        "create the loss object, "
                        "e.g. CategoricalCrossentropy(names=['dog', 'cat'])"
                    )
                    raise ValueError(msg)
                for i, value in enumerate(truths):
                    if value == missing_value:
                        truths[i] = self.names[0]
                        missing.append(i)
                truths = [self._name_to_i[name] for name in truths]
            truths = xp.asarray(truths, dtype="i")
        else:
            missing = []
        if truths.ndim != guesses.ndim:
            # transform categorical values to one-hot encoding
            truths = to_categorical(cast(Ints1d, truths), n_classes=guesses.shape[-1])
        mask = _make_mask(missing, guesses)
        return truths, mask

    def __call__(
        self, guesses: Floats2d, truths: IntsOrFloatsOrStrs
    ) -> Tuple[Floats2d, float]:
        d_truth = self.get_grad(guesses, truths)
        return (d_truth, self._get_loss_from_grad(d_truth))

    def get_grad(self, guesses: Floats2d, truths: IntsOrFloatsOrStrs) -> Floats2d:
        target, mask = self.convert_truths(truths, guesses)
        if guesses.shape != target.shape:  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss: mismatched shapes: {guesses.shape} vs {target.shape}."
            raise ValueError(err)
        if guesses.any() > 1 or guesses.any() < 0:  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss with guesses outside the [0,1] interval."
            raise ValueError(err)
        if target.any() > 1 or target.any() < 0:  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss with truth values outside the [0,1] interval."
            raise ValueError(err)
        difference = guesses - target
        difference *= mask
        if self.normalize:
            difference = difference / guesses.shape[0]
        return difference

    def get_loss(self, guesses: Floats2d, truths: IntsOrFloats) -> float:
        d_truth = self.get_grad(guesses, truths)
        return self._get_loss_from_grad(d_truth)

    def _get_loss_from_grad(self, d_truth: Floats2d) -> float:
        # TODO: Add overload for axis=None case to sum
        return (d_truth ** 2).sum()  # type: ignore


@registry.losses("CategoricalCrossentropy.v1")
def configure_CategoricalCrossentropy(
    *,
    normalize: bool = True,
    names: Optional[List[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
) -> CategoricalCrossentropy:
    return CategoricalCrossentropy(
        normalize=normalize, names=names, missing_value=missing_value
    )


class SequenceCategoricalCrossentropy(Loss):
    def __init__(
        self,
        *,
        normalize: bool = True,
        names: Optional[List[str]] = None,
        missing_value: Optional[Union[str, int]] = None,
    ):
        self.cc = CategoricalCrossentropy(
            normalize=False, names=names, missing_value=missing_value
        )
        self.normalize = normalize

    def __call__(
        self, guesses: List[Floats2d], truths: List[Union[Ints1d, Floats2d]]
    ) -> Tuple[List[Floats2d], float]:
        grads = self.get_grad(guesses, truths)
        loss = self._get_loss_from_grad(grads)
        return grads, loss

    def get_grad(
        self, guesses: List[Floats2d], truths: List[Union[Ints1d, Floats2d]]
    ) -> List[Floats2d]:
        err = "Cannot calculate SequenceCategoricalCrossentropy loss: guesses and truths must be same length"
        if len(guesses) != len(truths):  # pragma: no cover
            raise ValueError(err)
        n = len(guesses)
        d_scores = []
        for yh, y in zip(guesses, truths):
            d_yh = self.cc.get_grad(yh, y)
            if self.normalize:
                d_yh /= n
            d_scores.append(d_yh)
        return d_scores

    def get_loss(
        self, guesses: List[Floats2d], truths: List[Union[Ints1d, Floats2d]]
    ) -> float:
        return self._get_loss_from_grad(self.get_grad(guesses, truths))

    def _get_loss_from_grad(self, grads: List[Floats2d]) -> float:
        loss = 0.0
        for grad in grads:
            loss += self.cc._get_loss_from_grad(grad)
        return loss


@registry.losses("SequenceCategoricalCrossentropy.v1")
def configure_SequenceCategoricalCrossentropy(
    *, normalize: bool = True, names: Optional[List[str]] = None
) -> SequenceCategoricalCrossentropy:
    return SequenceCategoricalCrossentropy(normalize=normalize, names=names)


class L2Distance(Loss):
    def __init__(self, *, normalize: bool = True):
        self.normalize = normalize

    def __call__(self, guesses: Floats2d, truths: Floats2d) -> Tuple[Floats2d, float]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_grad(self, guesses: Floats2d, truths: Floats2d) -> Floats2d:
        if guesses.shape != truths.shape:  # pragma: no cover
            err = f"Cannot calculate L2 distance: mismatched shapes: {guesses.shape} vs {truths.shape}."
            raise ValueError(err)
        difference = guesses - truths
        if self.normalize:
            difference = difference / guesses.shape[0]
        return difference

    def get_loss(self, guesses: Floats2d, truths: Floats2d) -> float:
        if guesses.shape != truths.shape:  # pragma: no cover
            err = f"Cannot calculate L2 distance: mismatched shapes: {guesses.shape} vs {truths.shape}."
            raise ValueError(err)
        d_truth = self.get_grad(guesses, truths)
        # TODO: Add overload for axis=None case to sum
        return (d_truth ** 2).sum()  # type: ignore


@registry.losses("L2Distance.v1")
def configure_L2Distance(*, normalize: bool = True) -> L2Distance:
    return L2Distance(normalize=normalize)


class CosineDistance(Loss):
    def __init__(self, *, normalize: bool = True, ignore_zeros: bool = False):
        self.normalize = normalize
        self.ignore_zeros = ignore_zeros

    def __call__(self, guesses: Floats2d, truths: Floats2d) -> Tuple[Floats2d, float]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_similarity(self, guesses: Floats2d, truths: Floats2d) -> float:
        if guesses.shape != truths.shape:  # pragma: no cover
            err = f"Cannot calculate cosine similarity: mismatched shapes: {guesses.shape} vs {truths.shape}."
            raise ValueError(err)

        xp = get_array_module(guesses)
        # Add a small constant to avoid 0 vectors
        yh = guesses + 1e-8
        y = truths + 1e-8
        norm_yh = xp.linalg.norm(yh, axis=1, keepdims=True)
        norm_y = xp.linalg.norm(y, axis=1, keepdims=True)
        mul_norms = norm_yh * norm_y
        cosine = (yh * y).sum(axis=1, keepdims=True) / mul_norms
        return cosine

    def get_grad(self, guesses: Floats2d, truths: Floats2d) -> Floats2d:
        if guesses.shape != truths.shape:  # pragma: no cover
            err = f"Cannot calculate cosine similarity: mismatched shapes: {guesses.shape} vs {truths.shape}."
            raise ValueError(err)

        # Note: not using get_distance() here to avoid duplicating certain calculations
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
        if self.ignore_zeros:
            # If the target was a zero vector, don't count it in the loss.
            d_yh[zero_indices] = 0
        if self.normalize:
            d_yh = d_yh / guesses.shape[0]
        return -d_yh

    def get_loss(self, guesses: Floats2d, truths: Floats2d) -> float:
        if guesses.shape != truths.shape:  # pragma: no cover
            err = f"Cannot calculate cosine similarity: mismatched shapes: {guesses.shape} vs {truths.shape}."
            raise ValueError(err)

        xp = get_array_module(guesses)
        cosine = self.get_similarity(guesses, truths)
        losses = xp.abs(cosine - 1)
        if self.ignore_zeros:
            # If the target was a zero vector, don't count it in the loss.
            zero_indices = xp.abs(truths).sum(axis=1) == 0
            losses[zero_indices] = 0
        if self.normalize:
            losses = losses / guesses.shape[0]
        loss = losses.sum()
        return loss


@registry.losses("CosineDistance.v1")
def configure_CosineDistance(
    *, normalize: bool = True, ignore_zeros: bool = False
) -> CosineDistance:
    return CosineDistance(normalize=normalize, ignore_zeros=ignore_zeros)


def _make_mask(missing, guesses) -> Floats2d:
    xp = get_array_module(guesses)
    mask = xp.ones(guesses.shape, dtype="f")
    mask[missing] = 0
    return mask


__all__ = [
    "SequenceCategoricalCrossentropy",
    "CategoricalCrossentropy",
    "L2Distance",
    "CosineDistance",
]
