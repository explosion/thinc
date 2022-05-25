from ..util import to_categorical, get_array_module
from typing import Optional, Sequence, Dict, Union, Tuple
from typing import TypeVar, Generic, Any, cast, List
from ..types import Floats2d, Ints1d


LossT = TypeVar("LossT")
GradT = TypeVar("GradT")
GuessT = TypeVar("GuessT")
TruthT = TypeVar("TruthT")
IntsOrFloats = Union[Ints1d, Floats2d]
IntsOrFloatsOrStrs = Union[Ints1d, Floats2d, Sequence[int], Sequence[str]]


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

    def convert_truths(self, truths, guesses: Floats2d) -> Tuple[Floats2d, Floats2d]:
        xp = get_array_module(guesses)
        missing = []
        negatives_mask = None
        if self.names:
            negatives_mask = xp.ones((len(truths), len(self.names)), dtype="f")
        missing_value = self.missing_value
        # Convert list of ints or list of strings
        if isinstance(truths, list):
            truths = list(truths)
            if len(truths):
                if isinstance(truths[0], int):
                    for i, value in enumerate(truths):
                        if value == missing_value:
                            missing.append(i)
                else:
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
                        elif (
                            value
                            and self.neg_prefix
                            and value.startswith(self.neg_prefix)
                        ):
                            truths[i] = value[len(self.neg_prefix) :]
                            neg_index = self._name_to_i[truths[i]]
                            negatives_mask[i] = 0  # type: ignore
                            negatives_mask[i][neg_index] = -1  # type: ignore
                    truths = [self._name_to_i[name] for name in truths]
            truths = xp.asarray(truths, dtype="i")
            mask = _make_mask(guesses, missing)
        else:
            mask = _make_mask_by_value(truths, guesses, missing_value)
        if truths.ndim != guesses.ndim:
            # transform categorical values to one-hot encoding
            truths = to_categorical(
                cast(Ints1d, truths),
                n_classes=guesses.shape[-1],
                label_smoothing=self.label_smoothing,
            )
        else:
            if self.label_smoothing:
                raise ValueError(
                    "Label smoothing is only applied, when truths have type "
                    "List[str], List[int] or Ints1d, but it seems like Floats2d "
                    "was provided."
                )
        # Transform negative annotations to a 0 for the negated value
        # + mask all other values for that row
        if negatives_mask is not None:
            truths *= negatives_mask
            truths[truths == -1] = 0
            negatives_mask[negatives_mask == -1] = 1
            mask *= negatives_mask
        return truths, mask

    def __call__(
        self, guesses: Floats2d, truths: IntsOrFloatsOrStrs
    ) -> Tuple[Floats2d, float]:
        d_truth = self.get_grad(guesses, truths)
        return (d_truth, self._get_loss_from_grad(d_truth))

    def get_grad(self, guesses: Floats2d, truths: IntsOrFloatsOrStrs) -> Floats2d:
        target, mask = self.convert_truths(truths, guesses)
        xp = get_array_module(target)
        if guesses.shape != target.shape:  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss: mismatched shapes: {guesses.shape} vs {target.shape}."
            raise ValueError(err)
        if xp.any(guesses > 1) or xp.any(guesses < 0):  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss with guesses outside the [0,1] interval."
            raise ValueError(err)
        if xp.any(target > 1) or xp.any(target < 0):  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss with truth values outside the [0,1] interval."
            raise ValueError(err)
        difference = guesses - target
        difference *= mask
        if self.normalize:
            difference = difference / guesses.shape[0]
        return difference

    def get_loss(self, guesses: Floats2d, truths: IntsOrFloatsOrStrs) -> float:
        d_truth = self.get_grad(guesses, truths)
        return self._get_loss_from_grad(d_truth)

    def _get_loss_from_grad(self, d_truth: Floats2d) -> float:
        # TODO: Add overload for axis=None case to sum
        return (d_truth**2).sum()  # type: ignore


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
        grads = self.get_grad(guesses, truths)
        loss = self._get_loss_from_grad(grads)
        return grads, loss

    def get_grad(
        self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]
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
        self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]
    ) -> float:
        return self._get_loss_from_grad(self.get_grad(guesses, truths))

    def _get_loss_from_grad(self, grads: Sequence[Floats2d]) -> float:
        loss = 0.0
        for grad in grads:
            loss += self.cc._get_loss_from_grad(grad)
        return loss
