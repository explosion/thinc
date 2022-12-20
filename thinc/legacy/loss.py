from typing import Optional, Sequence, Dict, Union, Tuple
from typing import cast, List
from ..types import Floats2d, Ints1d
from ..config import registry
from ..util import to_categorical, get_array_module
from ..loss import IntsOrFloatsOrStrs, Loss
from ..loss import _make_mask, _make_mask_by_value


TruthsT = Union[List[Optional[str]], List[int], Ints1d, Floats2d]


class LegacyCategoricalCrossentropy(Loss):
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

    def convert_truths(
        self, truths: TruthsT, guesses: Floats2d
    ) -> Tuple[Floats2d, Floats2d]:
        xp = get_array_module(guesses)
        missing = []
        negatives_mask = None
        if self.names:
            negatives_mask = xp.ones((len(truths), len(self.names)), dtype="f")
        missing_value = self.missing_value
        # Convert list of ints or list of strings
        if isinstance(truths, list):
            if len(truths):
                if isinstance(truths[0], int):
                    for i, value in enumerate(truths):
                        if not isinstance(value, int):
                            raise ValueError(
                                "All values in the truths list have to "
                                "have the same type. The first value was "
                                f"detected to be integer, but found {type(value)}."
                            )
                        if value == missing_value:
                            missing.append(i)
                else:
                    truths = cast(List[Optional[str]], truths)
                    if self.names is None:
                        msg = (
                            "Cannot calculate loss from list of strings without names. "
                            "You can pass the names as a keyword argument when you "
                            "create the loss object, "
                            "e.g. CategoricalCrossentropy(names=['dog', 'cat'])"
                        )
                        raise ValueError(msg)
                    for i, value in enumerate(truths):
                        if not (isinstance(value, str) or value == missing_value):
                            raise ValueError(
                                "All values in the truths list have to "
                                "have the same type. The first value was "
                                f"detected to be string, but found {type(value)}."
                            )
                        if value == missing_value:
                            truths[i] = self.names[0]
                            missing.append(i)
                        elif (
                            value
                            and self.neg_prefix
                            and value.startswith(self.neg_prefix)
                        ):
                            neg_value = value[len(self.neg_prefix) :]
                            truths[i] = neg_value
                            neg_index = self._name_to_i[neg_value]
                            negatives_mask[i] = 0  # type: ignore
                            negatives_mask[i][neg_index] = -1  # type: ignore
                    # In the loop above, we have ensured that `truths` doesn't
                    # contain `None` (anymore). However, mypy can't infer this
                    # and doesn't like the shadowing.
                    truths_str = cast(List[str], truths)
                    truths = [self._name_to_i[name] for name in truths_str]
            truths = xp.asarray(truths, dtype="i")
            mask = _make_mask(guesses, missing)
        else:
            mask = _make_mask_by_value(truths, guesses, missing_value)
        truths = cast(Union[Ints1d, Floats2d], truths)
        if truths.ndim != guesses.ndim:
            # transform categorical values to one-hot encoding
            truths_2d = to_categorical(
                truths,
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
            truths_2d = cast(Floats2d, truths)
        # Transform negative annotations to a 0 for the negated value
        # + mask all other values for that row
        if negatives_mask is not None:
            truths_2d *= negatives_mask
            truths_2d[truths_2d == -1] = 0
            negatives_mask[negatives_mask == -1] = 1
            mask *= negatives_mask
        return cast(Floats2d, truths_2d), mask

    def __call__(self, guesses: Floats2d, truths: TruthsT) -> Tuple[Floats2d, float]:
        d_truth = self.get_grad(guesses, truths)
        return (d_truth, self._get_loss_from_grad(d_truth))

    def get_grad(self, guesses: Floats2d, truths: TruthsT) -> Floats2d:
        target, mask = self.convert_truths(truths, guesses)
        xp = get_array_module(target)
        if guesses.shape != target.shape:  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss: mismatched shapes: {guesses.shape} vs {target.shape}."
            raise ValueError(err)
        elif xp.any(guesses > 1) or xp.any(guesses < 0):  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss with guesses outside the [0,1] interval."
            raise ValueError(err)
        elif xp.any(target > 1) or xp.any(target < 0):  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss with truth values outside the [0,1] interval."
            raise ValueError(err)
        difference = guesses - target
        difference *= mask
        if self.normalize:
            difference = difference / guesses.shape[0]
        return difference

    def get_loss(self, guesses: Floats2d, truths: TruthsT) -> float:
        d_truth = self.get_grad(guesses, truths)
        return self._get_loss_from_grad(d_truth)

    def _get_loss_from_grad(self, d_truth: Floats2d) -> float:
        # TODO: Add overload for axis=None case to sum
        return (d_truth**2).sum()  # type: ignore


class LegacySequenceCategoricalCrossentropy(Loss):
    def __init__(
        self,
        *,
        normalize: bool = True,
        names: Optional[Sequence[str]] = None,
        missing_value: Optional[Union[str, int]] = None,
        neg_prefix: Optional[str] = None,
        label_smoothing: float = 0.0,
    ):
        self.cc = LegacyCategoricalCrossentropy(
            normalize=False,
            names=names,
            missing_value=missing_value,
            neg_prefix=neg_prefix,
            label_smoothing=label_smoothing,
        )
        self.normalize = normalize

    def __call__(
        self, guesses: Sequence[Floats2d], truths: Sequence[TruthsT]
    ) -> Tuple[List[Floats2d], float]:
        grads = self.get_grad(guesses, truths)
        loss = self._get_loss_from_grad(grads)
        return grads, loss

    def get_grad(
        self, guesses: Sequence[Floats2d], truths: Sequence[TruthsT]
    ) -> List[Floats2d]:
        if len(guesses) != len(truths):  # pragma: no cover
            err = "Cannot calculate SequenceCategoricalCrossentropy loss: guesses and truths must be same length"
            raise ValueError(err)
        n = len(guesses)
        d_scores = []
        for yh, y in zip(guesses, truths):
            d_yh = self.cc.get_grad(yh, y)
            if self.normalize:
                d_yh /= n
            d_scores.append(d_yh)
        return d_scores

    def get_loss(self, guesses: Sequence[Floats2d], truths: Sequence[TruthsT]) -> float:
        return self._get_loss_from_grad(self.get_grad(guesses, truths))

    def _get_loss_from_grad(self, grads: Sequence[Floats2d]) -> float:
        loss = 0.0
        for grad in grads:
            loss += self.cc._get_loss_from_grad(grad)  # type: ignore
        return loss


@registry.losses("CategoricalCrossentropy.v1")
def configure_CategoricalCrossentropy_v1(
    *,
    normalize: bool = True,
    names: Optional[Sequence[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
) -> LegacyCategoricalCrossentropy:
    return LegacyCategoricalCrossentropy(
        normalize=normalize, names=names, missing_value=missing_value
    )


@registry.losses("CategoricalCrossentropy.v2")
def configure_CategoricalCrossentropy_v2(
    *,
    normalize: bool = True,
    names: Optional[Sequence[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
    neg_prefix: Optional[str] = None,
) -> LegacyCategoricalCrossentropy:
    return LegacyCategoricalCrossentropy(
        normalize=normalize,
        names=names,
        missing_value=missing_value,
        neg_prefix=neg_prefix,
    )


@registry.losses("CategoricalCrossentropy.v3")
def configure_CategoricalCrossentropy_v3(
    *,
    normalize: bool = True,
    names: Optional[Sequence[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
    neg_prefix: Optional[str] = None,
    label_smoothing: float = 0.0,
) -> LegacyCategoricalCrossentropy:
    return LegacyCategoricalCrossentropy(
        normalize=normalize,
        names=names,
        missing_value=missing_value,
        neg_prefix=neg_prefix,
        label_smoothing=label_smoothing,
    )


@registry.losses("SequenceCategoricalCrossentropy.v1")
def configure_SequenceCategoricalCrossentropy_v1(
    *, normalize: bool = True, names: Optional[Sequence[str]] = None
) -> LegacySequenceCategoricalCrossentropy:
    return LegacySequenceCategoricalCrossentropy(normalize=normalize, names=names)


@registry.losses("SequenceCategoricalCrossentropy.v2")
def configure_SequenceCategoricalCrossentropy_v2(
    *,
    normalize: bool = True,
    names: Optional[Sequence[str]] = None,
    neg_prefix: Optional[str] = None,
) -> LegacySequenceCategoricalCrossentropy:
    return LegacySequenceCategoricalCrossentropy(
        normalize=normalize, names=names, neg_prefix=neg_prefix
    )


@registry.losses("SequenceCategoricalCrossentropy.v3")
def configure_SequenceCategoricalCrossentropy_v3(
    *,
    normalize: bool = True,
    names: Optional[Sequence[str]] = None,
    missing_value: Optional[Union[str, int]] = None,
    neg_prefix: Optional[str] = None,
    label_smoothing: float = 0.0,
) -> LegacySequenceCategoricalCrossentropy:
    return LegacySequenceCategoricalCrossentropy(
        normalize=normalize,
        names=names,
        neg_prefix=neg_prefix,
        missing_value=missing_value,
        label_smoothing=label_smoothing,
    )
