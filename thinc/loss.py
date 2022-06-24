from typing import Tuple, Sequence, cast, TypeVar, Generic, Any, Union, Optional, List
from typing import Dict

from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical, smooth_one_hot
from .util import is_xp_array
from .config import registry

LossT = TypeVar("LossT")
GradT = TypeVar("GradT")
GuessT = TypeVar("GuessT")
TruthT = TypeVar("TruthT")
IntsOrFloats = Union[Ints1d, Floats2d]
IntsOrFloatsOrStrs = Union[Ints1d, Floats2d, Sequence[int], Sequence[str]]
Categories1d = Union[Ints1d, Sequence[int], Sequence[str]]


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
    missing_value: Optional[Union[str, int]]

    def __init__(
        self,
        *,
        normalize: bool = True,
        missing_value: Optional[int] = None,
        label_smoothing: float = 0.0,
    ):
        self.normalize = normalize
        self.missing_value = missing_value
        self.label_smoothing = label_smoothing

    def convert_truths(
        self, truths: Floats2d, guesses: Floats2d
    ) -> Tuple[Floats2d, Floats2d]:
        missing_value = self.missing_value
        xp = get_array_module(guesses)
        mask = _make_mask_by_value(truths, guesses, missing_value)
        if not xp.allclose(truths.sum(axis=1), 1.0):
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy. "
                "All rows of 'truths' have to be a "
                "valid categorical distribution (sum to 1)."
            )
        if self.label_smoothing:
            # Validate that array is binary, ergo one-hot at this point
            if ((truths == 0) | (truths == 1)).all():
                truths = smooth_one_hot(truths, self.label_smoothing)
            else:
                raise ValueError(
                    "Can only apply label-smoothing to one-hot target."
                )
        return truths, mask

    def __call__(
        self, guesses: Floats2d, truths: IntsOrFloatsOrStrs
    ) -> Tuple[Floats2d, float]:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        d_truth = self._get_grad(guesses, target, mask)
        loss = self._get_loss(guesses, target, mask)
        return (d_truth, loss)

    def _validate_input(self, guesses: Floats2d, target: Floats2d) -> None:
        xp = get_array_module(target)
        if not xp.allclose(guesses.sum(axis=1), 1.0):
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy if "
                "some rows of 'guesses' are not "
                "valid categorical distributions (do not sum to 1)."
            )
        if guesses.shape != target.shape:  # pragma: no cover
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy loss: "
                f"mismatched shapes: {guesses.shape} vs {target.shape}."
            )
        if xp.any(guesses > 1) or xp.any(guesses < 0):  # pragma: no cover
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy loss "
                "with guesses outside the [0,1] interval."
            )
        if xp.any(target > 1) or xp.any(target < 0):  # pragma: no cover
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy loss "
                "with truth values outside the [0,1] interval."
            )

    def _get_grad(
        self, guesses: Floats2d, target: Floats2d, mask: Floats2d
    ) -> Floats2d:
        difference = guesses - target
        difference *= mask
        if self.normalize:
            difference = difference / guesses.shape[0]
        return cast(Floats2d, difference.astype("float32"))

    def _get_loss(self, guesses: Floats2d, target: Floats2d, mask: Floats2d) -> float:
        xp = get_array_module(guesses)
        logprobs = xp.log(guesses + 1e-9)
        logprobs *= mask
        if self.normalize:
            return -(target * logprobs).sum(1).mean()
        else:
            return -(target * logprobs).sum()

    def get_grad(self, guesses: Floats2d, truths: IntsOrFloatsOrStrs) -> Floats2d:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        return self._get_grad(guesses, target, mask)

    def get_loss(self, guesses: Floats2d, truths: IntsOrFloatsOrStrs) -> float:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        return self._get_loss(guesses, target, mask)


class SparseCE(CategoricalCrossentropy):
    names: Optional[Sequence[str]]
    missing_value: Optional[Union[str, int]]
    _name_to_i: Dict[str, int]

    def __init__(
        self,
        *,
        normalize: bool = True,
        names: Optional[Sequence[str]] = None,
        missing_value: Optional[Union[str, int]] = None,
        neg_prefix: Optional[str] = None,
        label_smoothing: float = 0.0,
    ):
        self.normalize = normalize
        self.names = names
        self.missing_value = missing_value
        self.neg_prefix = neg_prefix
        self.label_smoothing = label_smoothing
        if names is not None:
            self._name_to_i = {name: i for i, name in enumerate(names)}
        else:
            self._name_to_i = {}

    def _check_ints1d(self, arr):
        """
        Check whether array is 1D and has type integer.
        """
        if arr.ndim != 1:
            raise ValueError(
                "SparseCE only accepts 1D arrays, but "
                f"array with shape {arr.shape} was given."
            )
        if arr.dtype.kind != 'i':
            raise ValueError(
                "SparseCE only accepts integer arrays, but "
                f"array with {arr.dtype} was given."
            )

    def _convert_ints(
        self, guesses: Floats2d, truths: Sequence[int]
    ) -> List[int]:
        """
        Convert Sequence[int] into a Floats2d one-hot array.
        """
        missing_value = self.missing_value
        if missing_value is not None and not isinstance(missing_value, int):
            raise ValueError(
                "truths provided in Sequence[int] format, but "
                f"missing_value was set to be {self.missing_value} "
                f", which has type {type(self.missing_value)}."
            )
        missing = []
        for i, value in enumerate(truths):
            if not isinstance(value, int):
                raise ValueError(
                    "The first value of the truths was of type "
                    f"integer, but found {type(value)} during iteration."
                )
            if value == missing_value:
                missing.append(i)
        xp = get_array_module(guesses)
        truths = xp.asarray(truths, dtype="i")
        truths = to_categorical(
            truths, n_classes=guesses.shape[-1], label_smoothing=self.label_smoothing
        )
        mask = _make_mask(guesses, missing)
        return truths, mask

    def _convert_strs(
        self, guesses: Floats2d, truths: Sequence[str]
    ):
        """
        Convert Sequence[int] into a Floats2d one-hot array.
        """

        missing_value = self.missing_value
        if self.names is None:
            raise ValueError(
                "Cannot calculate loss from Sequence[str] without names. "
                "You can pass the names as a keyword argument when you "
                "create the loss object, "
                "e.g. CategoricalCrossentropy(names=['dog', 'cat'])"
            )
        if missing_value is not None and not isinstance(missing_value, str):
            raise ValueError(
                "truths provided in Sequence[str] format, but "
                f"missing_value was set to be {self.missing_value} "
                f", which has type {type(self.missing_value)}."
            )
        xp = get_array_module(guesses)
        missing = []
        negatives_mask = xp.ones((len(truths), len(self.names)), dtype="f")
        truths_int = []
        for i, value in enumerate(truths):
            if not isinstance(value, str):
                raise ValueError(
                    "The first value of the truths was of type "
                    f"string, but found {type(value)} during iteration."
                )
            # missing value
            if value == missing_value:
                label_i = self._name_to_i[self.names[0]]
                missing.append(i)
            # negative labels
            elif (
                self.neg_prefix
                and value.startswith(self.neg_prefix)
            ):
                label_i = self._name_to_i[value[len(self.neg_prefix) :]]
                negatives_mask[i] = 0  # type: ignore
                negatives_mask[i][label_i] = -1  # type: ignore
            # nothing special
            else:
                label_i = self._name_to_i[value]
            truths_int.append(label_i)
        truths = xp.asarray(truths_int, dtype="i")
        truths = to_categorical(
            truths, n_classes=guesses.shape[-1], label_smoothing=self.label_smoothing
        )
        mask = _make_mask(guesses, missing)
        truths *= negatives_mask
        truths[truths == -1] = 0
        negatives_mask[negatives_mask == -1] = 1
        mask *= negatives_mask
        return truths, mask

    def convert_truths(
        self, truths: Categories1d, guesses: Floats2d
    ) -> Tuple[Floats2d, Floats2d]:

        if is_xp_array(truths):
            self._check_ints1d(truths)
            truths = to_categorical(
                truths, label_smoothing=self.label_smoothing
            )
            mask = _make_mask_by_value(truths, guesses, self.missing_value)
        elif isinstance(truths, Sequence):
            if isinstance(truths[0], int):
                truths, mask = self._convert_ints(guesses, truths)
            elif isinstance(truths[0], str):
                truths, mask = self._convert_strs(guesses, truths)
            else:
                raise ValueError(
                    "When truths to SparseCE is provided "
                    "in Sequence format, elements need to be "
                    "of type str or int, but first element "
                    f"was found to be {type(truths[0])}."
                )
        else:
            raise ValueError(
                "Truths have to be provided either as 1D "
                "numpy/cupy integer array or as Sequence[int] or "
                "Sequence[str], but truths has different type."
            )

        return truths, mask


@registry.losses("CategoricalCrossentropy.v4")
def configure_CategoricalCrossentropy_v4(
    *,
    normalize: bool = True,
    names: Optional[Sequence[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
    neg_prefix: Optional[str] = None,
    label_smoothing: float = 0.0,
    sparse: bool = True,
) -> CategoricalCrossentropy:
    if names is None and neg_prefix is None and not sparse:
        return CategoricalCrossentropy(
            normalize=normalize,
            missing_value=missing_value,
            label_smoothing=label_smoothing,
        )
    else:
        return SparseCE(
            normalize=normalize,
            names=names,
            missing_value=missing_value,
            neg_prefix=neg_prefix,
            label_smoothing=label_smoothing
        )


class SequenceCategoricalCrossentropy(Loss):
    def __init__(
        self,
        *,
        normalize: bool = True,
        names: Optional[Sequence[str]] = None,
        missing_value: Optional[Union[str, int]] = None,
        neg_prefix: Optional[str] = None,
        label_smoothing: float = 0.0,
    ):
        self.cc = CategoricalCrossentropy(
            normalize=False,
            names=names,
            missing_value=missing_value,
            neg_prefix=neg_prefix,
            label_smoothing=label_smoothing,
        )
        self.normalize = normalize

    def __call__(
        self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]
    ) -> Tuple[List[Floats2d], float]:
        self._validate_input(guesses, truths)
        n = len(guesses)
        d_scores = []
        loss = 0.0
        for yh, y in zip(guesses, truths):
            d_yh, l = self.cc(yh, y)
            if self.normalize:
                d_yh /= n
            d_scores.append(d_yh)
            loss += l
        return d_scores, loss

    def _validate_input(
        self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]
    ):
        if len(guesses) != len(truths):  # pragma: no cover
            raise ValueError(
                "Cannot calculate SequenceCategoricalCrossentropy loss: "
                "guesses and truths must be same length!"
            )

    def get_grad(
        self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]
    ) -> List[Floats2d]:
        self._validate_input(guesses, truths)
        n = len(guesses)
        d_scores = []
        for yh, y in zip(guesses, truths):
            d_yh = self.cc.get_grad(yh, y)
            if self.normalize:
                d_yh /= n
            d_scores.append(d_yh)
        return d_scores

    def get_loss(
        self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]
    ) -> float:
        self._validate_input(guesses, truths)
        loss = 0.0
        for guess, truth in zip(guesses, truths):
            loss += self.cc.get_loss(guess, truth)
        return loss


@registry.losses("SequenceCategoricalCrossentropy.v4")
def configure_SequenceCategoricalCrossentropy_v4(
    *,
    normalize: bool = True,
    names: Optional[Sequence[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
    neg_prefix: Optional[str] = None,
    label_smoothing: float = 0.0,
) -> SequenceCategoricalCrossentropy:
    return SequenceCategoricalCrossentropy(
        normalize=normalize,
        names=names,
        missing_value=missing_value,
        neg_prefix=neg_prefix,
        label_smoothing=label_smoothing
    )


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
        return (d_truth**2).sum()  # type: ignore


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
        d_yh = (y / mul_norms) - (cosine * (yh / norm_yh**2))
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


def _make_mask(guesses, missing) -> Floats2d:
    xp = get_array_module(guesses)
    mask = xp.ones(guesses.shape, dtype="f")
    mask[missing] = 0
    return mask


def _make_mask_by_value(truths, guesses, missing_value) -> Floats2d:
    xp = get_array_module(guesses)
    mask = xp.ones(guesses.shape, dtype="f")

    if missing_value is not None:
        if truths.ndim == 1:
            mask[truths == missing_value] = 0.0
        else:
            # In 2D truths, labels are encoded as one-hot vectors, so we can get
            # the label indices using argmax.
            labels = xp.argmax(truths, axis=-1)
            mask[labels == missing_value] = 0.0

    return mask


__all__ = [
    "SequenceCategoricalCrossentropy",
    "CategoricalCrossentropy",
    "L2Distance",
    "CosineDistance",
]
