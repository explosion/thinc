from typing import Tuple, Sequence, cast, TypeVar, Generic, Any, Union, Optional, List
from typing import Dict
from abc import abstractmethod

from .types import Floats2d, Ints1d, Ragged, ArrayXd
from .util import get_array_module, to_categorical, smooth_one_hot
from .util import is_xp_array
from .config import registry

LossT = TypeVar("LossT")
GradT = TypeVar("GradT")
GuessT = TypeVar("GuessT")
TruthT = TypeVar("TruthT")
FloatsOrRaggedT = TypeVar("FloatsOrRaggedT", Floats2d, Ragged)
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

    @abstractmethod
    def get_grad(self, guesses: GuessT, truths: TruthT) -> GradT:
        ...

    @abstractmethod
    def get_loss(self, guesses: GuessT, truths: TruthT) -> LossT:
        ...


class CategoricalCrossentropyBase(Loss):
    normalize: bool

    def _validate_input(self, guesses: FloatsOrRaggedT, target: Floats2d) -> None:
        guesses_f2d = _to_array(guesses)
        xp = get_array_module(target)
        if not xp.allclose(guesses_f2d.sum(axis=1), 1.0):
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy if "
                "some rows of 'guesses' are not "
                "valid categorical distributions (do not sum to 1)."
            )
        elif guesses_f2d.shape != target.shape:  # pragma: no cover
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy loss "
                f"with mismatching shapes: {guesses_f2d.shape} vs {target.shape}."
            )
        elif xp.any(guesses_f2d > 1) or xp.any(guesses_f2d < 0):  # pragma: no cover
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy loss "
                "with guesses outside the [0,1] interval."
            )
        elif xp.any(target > 1) or xp.any(target < 0):  # pragma: no cover
            raise ValueError(
                "Cannot calculate CategoricalCrossentropy loss "
                "with truth values outside the [0,1] interval."
            )

    def _get_grad(
        self, guesses: FloatsOrRaggedT, target: Floats2d, mask: Floats2d
    ) -> FloatsOrRaggedT:
        difference = _to_array(guesses) - target
        difference *= mask
        if self.normalize:
            # FIXME: normalized by the number of sequences, also support normalizing
            #  by the number of instances.
            difference /= _normalization_length(guesses)

        return _array_like(difference, guesses)

    def _get_loss(
        self, guesses: FloatsOrRaggedT, target: Floats2d, mask: Floats2d
    ) -> float:
        guesses_f2d = _to_array(guesses)
        xp = get_array_module(guesses_f2d)
        logprobs = xp.log(guesses_f2d + 1e-9)
        logprobs *= mask
        if self.normalize:
            return -(target * logprobs).sum() / _normalization_length(guesses)
        else:
            return -(target * logprobs).sum()


class CategoricalCrossentropy(CategoricalCrossentropyBase):
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

    def __call__(
        self, guesses: FloatsOrRaggedT, truths: Floats2d
    ) -> Tuple[FloatsOrRaggedT, float]:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        d_truth = self._get_grad(guesses, target, mask)
        loss = self._get_loss(guesses, target, mask)

        return d_truth, loss

    def convert_truths(
        self, truths: Floats2d, guesses: FloatsOrRaggedT
    ) -> Tuple[Floats2d, Floats2d]:
        if truths.ndim != 2:
            raise ValueError(f"'truths' have to have 2 axes, but found {truths.ndim}")
        guesses_2d = _to_array(guesses)
        missing_value = self.missing_value
        xp = get_array_module(guesses_2d)
        mask = _make_mask_by_value(truths, guesses_2d, missing_value)
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
                raise ValueError("Can only apply label-smoothing to one-hot target.")
        return truths, mask

    def get_grad(self, guesses: FloatsOrRaggedT, truths: Floats2d) -> FloatsOrRaggedT:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        return self._get_grad(guesses, target, mask)

    def get_loss(self, guesses: Floats2d, truths: Floats2d) -> float:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        return self._get_loss(guesses, target, mask)


class SparseCategoricalCrossentropy(CategoricalCrossentropyBase):
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

    def __call__(
        self, guesses: Floats2d, truths: Union[Sequence[int], Sequence[str]]
    ) -> Tuple[Floats2d, float]:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        d_truth = self._get_grad(guesses, target, mask)
        loss = self._get_loss(guesses, target, mask)
        return (d_truth, loss)

    def _convert_ints(
        self, guesses: Floats2d, truths: Sequence[int]
    ) -> Tuple[Floats2d, Floats2d]:
        """
        Convert Sequence[int] into a Floats2d one-hot array.
        """
        missing_value = self.missing_value
        if missing_value is not None and not isinstance(missing_value, int):
            raise ValueError(
                "'truths' provided in Sequence[int] format, but "
                f"'missing_value' was set to be {self.missing_value} "
                f", which has type {type(self.missing_value)}."
            )
        missing = []
        for i, value in enumerate(truths):
            if not isinstance(value, int):
                raise ValueError(
                    "The first value of `truths` was of type "
                    f"integer, but found {type(value)} during iteration."
                )
            if value == missing_value:
                missing.append(i)
        xp = get_array_module(guesses)
        # FIXME: convert using ops?
        xp_truths = cast(Ints1d, xp.asarray(truths, dtype="i"))
        truths_2d = to_categorical(
            xp_truths, n_classes=guesses.shape[-1], label_smoothing=self.label_smoothing
        )
        mask = _make_mask(guesses, missing)
        return cast(Floats2d, truths_2d), mask

    def _convert_strs(
        self, guesses: Floats2d, truths: Sequence[str]
    ) -> Tuple[Floats2d, Floats2d]:
        """
        Convert Sequence[int] into a Floats2d one-hot array.
        """

        missing_value = self.missing_value
        if self.names is None:
            raise ValueError(
                "Cannot calculate loss from Sequence[str] without names. "
                "You can pass the names as a keyword argument when you "
                "create the loss object"
            )
        elif missing_value is not None and not isinstance(missing_value, str):
            raise ValueError(
                "'truths' provided in Sequence[str] format, but "
                f"'missing_value' was set to be {self.missing_value} "
                f", which has type {type(self.missing_value)}."
            )
        xp = get_array_module(guesses)
        missing = []
        negatives_mask = xp.ones((len(truths), len(self.names)), dtype="f")
        truths_int = []
        for i, value in enumerate(truths):
            if not isinstance(value, str):
                raise ValueError(
                    "The first value of the 'truths' was of type "
                    f"string, but found {type(value)} during iteration."
                )
            # missing value
            if value == missing_value:
                label_i = self._name_to_i[self.names[0]]
                missing.append(i)
            # negative labels
            elif self.neg_prefix and value.startswith(self.neg_prefix):
                label_i = self._name_to_i[value[len(self.neg_prefix) :]]
                negatives_mask[i] = 0  # type: ignore
                negatives_mask[i][label_i] = -1  # type: ignore
            # nothing special
            else:
                label_i = self._name_to_i[value]
            truths_int.append(label_i)
        xp_truths = cast(Ints1d, xp.asarray(truths_int, dtype="i"))
        truths_2d = to_categorical(
            xp_truths, n_classes=guesses.shape[-1], label_smoothing=self.label_smoothing
        )
        mask = _make_mask(guesses, missing)
        truths_2d *= negatives_mask
        truths_2d[truths_2d == -1] = 0
        negatives_mask[negatives_mask == -1] = 1
        mask *= negatives_mask
        return cast(Floats2d, truths_2d), mask

    def convert_truths(
        self, truths: Categories1d, guesses: Floats2d
    ) -> Tuple[Floats2d, Floats2d]:
        guesses_f2d = _to_array(guesses)

        if is_xp_array(truths):
            _check_ints1d(cast(ArrayXd, truths))
            xp_truths = cast(Ints1d, truths)
            truths_2d = to_categorical(
                xp_truths,
                label_smoothing=self.label_smoothing,
                n_classes=guesses_f2d.shape[1],
            )
            mask = _make_mask_by_value(truths_2d, guesses_f2d, self.missing_value)
        elif isinstance(truths, Sequence):
            if isinstance(truths[0], int):
                truths_2d, mask = self._convert_ints(
                    guesses_f2d, cast(Sequence[int], truths)
                )
            elif isinstance(truths[0], str):
                truths_2d, mask = self._convert_strs(
                    guesses_f2d, cast(Sequence[str], truths)
                )
            else:
                raise ValueError(
                    "When truths to SparseCategoricalCrossentropy is provided "
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

        return cast(Floats2d, truths_2d), mask

    def get_grad(self, guesses: Floats2d, truths: Categories1d) -> Floats2d:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        return self._get_grad(guesses, target, mask)

    def get_loss(self, guesses: Floats2d, truths: Categories1d) -> float:
        target, mask = self.convert_truths(truths, guesses)
        self._validate_input(guesses, target)
        return self._get_loss(guesses, target, mask)


@registry.losses("CategoricalCrossentropy.v4")
def configure_CategoricalCrossentropy_v4(
    *,
    normalize: bool = True,
    missing_value: Optional[int] = None,
    label_smoothing: float = 0.0,
) -> CategoricalCrossentropy:
    return CategoricalCrossentropy(
        normalize=normalize,
        missing_value=missing_value,
        label_smoothing=label_smoothing,
    )


@registry.losses("SparseCategoricalCrossentropy.v4")
def configure_SparseCategoricalCrossentropy_v4(
    *,
    normalize: bool = True,
    names: Optional[Sequence[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
    neg_prefix: Optional[str] = None,
    label_smoothing: float = 0.0,
) -> SparseCategoricalCrossentropy:
    return SparseCategoricalCrossentropy(
        normalize=normalize,
        names=names,
        missing_value=missing_value,
        neg_prefix=neg_prefix,
        label_smoothing=label_smoothing,
    )


class SequenceCategoricalCrossentropy(Loss):
    def __init__(
        self,
        *,
        cross_entropy: Union[CategoricalCrossentropy, SparseCategoricalCrossentropy],
        normalize: bool = True,
    ):
        self.cc = cross_entropy
        self.normalize = normalize

    def __call__(
        self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]
    ) -> Tuple[List[Floats2d], float]:
        self._validate_input(guesses, truths)
        n = len(guesses)
        d_scores = []
        loss = 0.0
        for yh, y in zip(guesses, truths):
            d_yh, l = self.cc(yh, y)  # type: ignore
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
            d_yh = self.cc.get_grad(yh, y)  # type: ignore
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
            loss += self.cc.get_loss(guess, truth)  # type: ignore
        return loss


@registry.losses("SequenceCategoricalCrossentropy.v4")
def configure_SequenceCategoricalCrossentropy_v4(
    *,
    normalize: bool = True,
    sparse: bool = True,
    names: Optional[Sequence[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
    neg_prefix: Optional[str] = None,
    label_smoothing: float = 0.0,
) -> SequenceCategoricalCrossentropy:
    if names is None and neg_prefix is None and not sparse:
        cross_entropy: Union[
            CategoricalCrossentropy, SparseCategoricalCrossentropy
        ] = CategoricalCrossentropy(
            normalize=False,
            missing_value=cast(Optional[int], missing_value),
            label_smoothing=label_smoothing,
        )
    else:
        cross_entropy = SparseCategoricalCrossentropy(
            normalize=False,
            names=names,
            missing_value=cast(Optional[Union[str, int]], missing_value),
            neg_prefix=neg_prefix,
            label_smoothing=label_smoothing,
        )
    return SequenceCategoricalCrossentropy(
        cross_entropy=cross_entropy,
        normalize=normalize,
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


def _array_like(a: Floats2d, like: FloatsOrRaggedT) -> FloatsOrRaggedT:
    if isinstance(like, Ragged):
        return Ragged(a, lengths=like.lengths)
    else:
        return a


def _to_array(guesses: FloatsOrRaggedT) -> Floats2d:
    if isinstance(guesses, Ragged):
        return cast(Floats2d, guesses.data.astype("float32"))
    else:
        return guesses


def _normalization_length(guesses: FloatsOrRaggedT) -> int:
    if isinstance(guesses, Ragged):
        return len(guesses.lengths)
    else:
        return guesses.shape[0]


def _check_ints1d(arr: ArrayXd):
    """
    Check whether array is 1D and has type integer.
    """
    if arr.ndim != 1:
        raise ValueError(
            "SparseCategoricalCrossentropy only accepts 1D arrays, but "
            f"array with shape {arr.shape} was given."
        )
    if arr.dtype.kind != "i":  # type: ignore
        raise ValueError(
            "SparseCategoricalCrossentropy only accepts integer arrays, but "
            f"array with {arr.dtype} was given."
        )


__all__ = [
    "SequenceCategoricalCrossentropy",
    "CategoricalCrossentropy",
    "L2Distance",
    "CosineDistance",
]
