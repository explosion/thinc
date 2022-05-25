import pytest
import numpy
from thinc.legacy import CategoricalCrossentropy as LegacyCrossEntropy
from thinc.legacy import SequenceCategoricalCrossentropy as LegacySequenceCrossEntropy
from thinc import registry
from thinc import loss

# some simple arrays
scores0 = numpy.zeros((3, 3), dtype="f")
labels0 = numpy.asarray([0, 1, 1], dtype="i")

# a few more diverse ones to test realistic values
guesses1 = numpy.asarray([[0.1, 0.5, 0.6], [0.4, 0.6, 0.3], [1, 1, 1], [0, 0, 0]])
labels1 = numpy.asarray([2, 1, 0, 2])
labels1_full = numpy.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
labels1_strings = ["C", "B", "A", "C"]

guesses2 = numpy.asarray([[0.2, 0.3, 0.0]])
labels2 = numpy.asarray([1])
labels2_strings = ["B"]

eps = 0.0001


def test_loss():
    d_scores = LegacyCrossEntropy().get_grad(scores0, labels0)
    assert d_scores.dtype == "float32"
    assert d_scores.shape == scores0.shape
    d_scores = LegacySequenceCrossEntropy().get_grad([scores0], [labels0])
    assert d_scores[0].dtype == "float32"
    assert d_scores[0].shape == scores0.shape
    assert LegacySequenceCrossEntropy().get_grad([], []) == []


@pytest.mark.parametrize(
    "dist", [LegacyCrossEntropy()]
)
@pytest.mark.parametrize("vect", [scores0, guesses1, guesses2])
def test_equality(dist, vect):
    assert int(dist.get_grad(vect, vect)[0][0]) == pytest.approx(0, eps)
    assert dist.get_loss(vect, vect) == pytest.approx(0, eps)


@pytest.mark.parametrize(
    "guesses, labels", [(guesses1, labels1), (guesses1, labels1_full)]
)
def test_categorical_crossentropy(guesses, labels):
    d_scores = LegacyCrossEntropy(normalize=True).get_grad(guesses, labels)
    assert d_scores.shape == guesses.shape

    # The normalization divides the difference (e.g. 0.4) by the number of vectors (4)
    assert d_scores[1][0] == pytest.approx(0.1, eps)
    assert d_scores[1][1] == pytest.approx(-0.1, eps)

    # The third vector predicted all labels, but only the first one was correct
    assert d_scores[2][0] == pytest.approx(0, eps)
    assert d_scores[2][1] == pytest.approx(0.25, eps)
    assert d_scores[2][2] == pytest.approx(0.25, eps)

    # The fourth vector predicted no labels but should have predicted the last one
    assert d_scores[3][0] == pytest.approx(0, eps)
    assert d_scores[3][1] == pytest.approx(0, eps)
    assert d_scores[3][2] == pytest.approx(-0.25, eps)

    loss = LegacyCrossEntropy(normalize=True).get_loss(guesses, labels)
    assert loss == pytest.approx(0.239375, eps)


def test_crossentropy_incorrect_scores_targets():
    labels = numpy.asarray([2])

    guesses_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        LegacyCrossEntropy(normalize=True).get_grad(guesses_neg, labels)

    guesses_larger_than_one = numpy.asarray([[1.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        LegacyCrossEntropy(normalize=True).get_grad(
            guesses_larger_than_one, labels
        )

    guesses_ok = numpy.asarray([[0.1, 0.4, 0.5]])
    targets_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        LegacyCrossEntropy(normalize=True).get_grad(guesses_ok, targets_neg)

    targets_larger_than_one = numpy.asarray([[2.0, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        LegacyCrossEntropy(normalize=True).get_grad(
            guesses_ok, targets_larger_than_one
        )


@pytest.mark.parametrize(
    "guesses, labels",
    [(guesses1, [2, 1, 0, 2])],
)
def test_categorical_crossentropy_int_list_missing(guesses, labels):
    d_scores = LegacyCrossEntropy(normalize=True, missing_value=0).get_grad(
        guesses, labels
    )
    assert d_scores.shape == guesses.shape

    # The normalization divides the difference (e.g. 0.4) by the number of vectors (4)
    assert d_scores[1][0] == pytest.approx(0.1, eps)
    assert d_scores[1][1] == pytest.approx(-0.1, eps)

    # Label 0 is masked, because it represents the missing value
    assert d_scores[2][0] == 0.0
    assert d_scores[2][1] == 0.0
    assert d_scores[2][2] == 0.0

    # The fourth vector predicted no labels but should have predicted the last one
    assert d_scores[3][0] == pytest.approx(0, eps)
    assert d_scores[3][1] == pytest.approx(0, eps)
    assert d_scores[3][2] == pytest.approx(-0.25, eps)

    loss = LegacyCrossEntropy(normalize=True, missing_value=0).get_loss(
        guesses, labels
    )
    assert loss == pytest.approx(0.114375, eps)


@pytest.mark.parametrize(
    "guesses, labels", [(guesses1, labels1), (guesses1, labels1_full)]
)
def test_categorical_crossentropy_missing(guesses, labels):
    d_scores = LegacyCrossEntropy(normalize=True, missing_value=0).get_grad(
        guesses, labels
    )
    assert d_scores.shape == guesses.shape

    # The normalization divides the difference (e.g. 0.4) by the number of vectors (4)
    assert d_scores[1][0] == pytest.approx(0.1, eps)
    assert d_scores[1][1] == pytest.approx(-0.1, eps)

    # Label 0 is masked, because it represents the missing value
    assert d_scores[2][0] == 0.0
    assert d_scores[2][1] == 0.0
    assert d_scores[2][2] == 0.0

    # The fourth vector predicted no labels but should have predicted the last one
    assert d_scores[3][0] == pytest.approx(0, eps)
    assert d_scores[3][1] == pytest.approx(0, eps)
    assert d_scores[3][2] == pytest.approx(-0.25, eps)

    loss = LegacyCrossEntropy(normalize=True, missing_value=0).get_loss(
        guesses, labels
    )
    assert loss == pytest.approx(0.114375, eps)


@pytest.mark.parametrize(
    "guesses, labels, names",
    [
        ([guesses1, guesses2], [labels1, labels2], []),
        ([guesses1, guesses2], [labels1_full, labels2], []),
        ([guesses1, guesses2], [labels1_strings, labels2_strings], ["A", "B", "C"]),
    ],
)
def test_sequence_categorical_crossentropy(guesses, labels, names):
    d_scores = LegacySequenceCrossEntropy(normalize=False, names=names).get_grad(guesses, labels)
    d_scores1 = d_scores[0]
    d_scores2 = d_scores[1]
    assert d_scores1.shape == guesses1.shape
    assert d_scores2.shape == guesses2.shape
    assert d_scores1[1][0] == pytest.approx(0.4, eps)
    assert d_scores1[1][1] == pytest.approx(-0.4, eps)
    # The normalization divides the difference (e.g. 0.4) by the number of seqs
    d_scores = LegacySequenceCrossEntropy(normalize=True, names=names).get_grad(guesses, labels)
    d_scores1 = d_scores[0]
    d_scores2 = d_scores[1]

    assert d_scores1[1][0] == pytest.approx(0.2, eps)
    assert d_scores1[1][1] == pytest.approx(-0.2, eps)

    # The third vector predicted all labels, but only the first one was correct
    assert d_scores1[2][0] == pytest.approx(0, eps)
    assert d_scores1[2][1] == pytest.approx(0.5, eps)
    assert d_scores1[2][2] == pytest.approx(0.5, eps)

    # The fourth vector predicted no labels but should have predicted the last one
    assert d_scores1[3][0] == pytest.approx(0, eps)
    assert d_scores1[3][1] == pytest.approx(0, eps)
    assert d_scores1[3][2] == pytest.approx(-0.5, eps)

    # Test the second batch
    assert d_scores2[0][0] == pytest.approx(0.1, eps)
    assert d_scores2[0][1] == pytest.approx(-0.35, eps)

    loss = LegacySequenceCrossEntropy(normalize=True, names=names).get_loss(guesses, labels)
    assert loss == pytest.approx(1.09, eps)


@pytest.mark.parametrize(
    "guesses, labels, names",
    [
        ([guesses1], [["A", "!A", "", "!C"]], ["A", "B", "C"]),
    ],
)
def test_sequence_categorical_missing_negative(guesses, labels, names):
    d_scores = LegacySequenceCrossEntropy(
        normalize=False,
        names=names,
        neg_prefix="!",
        missing_value="",
    ).get_grad(guesses, labels)
    d_scores0 = d_scores[0]

    # [0.1, 0.5, 0.6] should be A
    assert d_scores0[0][0] == pytest.approx(-0.9, eps)
    assert d_scores0[0][1] == pytest.approx(0.5, eps)
    assert d_scores0[0][2] == pytest.approx(0.6, eps)

    # [0.4, 0.6, 0.3] should NOT be A
    assert d_scores0[1][0] == pytest.approx(0.4, eps)
    assert d_scores0[1][1] == pytest.approx(0.0, eps)
    assert d_scores0[1][2] == pytest.approx(0.0, eps)

    # [1, 1, 1] has missing gold label
    assert d_scores0[2][0] == pytest.approx(0.0, eps)
    assert d_scores0[2][1] == pytest.approx(0.0, eps)
    assert d_scores0[2][2] == pytest.approx(0.0, eps)

    # [0.0, 0.0, 0.0] should NOT be C
    assert d_scores0[3][0] == pytest.approx(0.0, eps)
    assert d_scores0[3][1] == pytest.approx(0.0, eps)
    assert d_scores0[3][2] == pytest.approx(0.0, eps)


@pytest.mark.parametrize(
    "name,kwargs,args",
    [
        ("CategoricalCrossentropy.v1", {}, (scores0, labels0)),
        ("SequenceCategoricalCrossentropy.v1", {}, ([scores0], [labels0])),
        ("CategoricalCrossentropy.v2", {"neg_prefix": "!"}, (scores0, labels0)),
        ("CategoricalCrossentropy.v3", {"neg_prefix": "!"}, (scores0, labels0)),
        ("SequenceCategoricalCrossentropy.v2", {"neg_prefix": "!"}, ([scores0], [labels0])),
        ("SequenceCategoricalCrossentropy.v3", {"neg_prefix": "!"}, ([scores0], [labels0])),
    ],
)
def test_loss_from_config(name, kwargs, args):
    """Test that losses are loaded and configured correctly from registry
    (as partials)."""
    cfg = {"test": {"@losses": name, **kwargs}}
    func = registry.resolve(cfg)["test"]
    loss = func.get_grad(*args)
    if isinstance(loss, (list, tuple)):
        loss = loss[0]
    assert loss.ndim == 2
    func.get_loss(*args)
    func(*args)
