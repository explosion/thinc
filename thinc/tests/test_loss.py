import pytest
import numpy
from functools import partial
from thinc.api import CategoricalCrossentropy
from thinc.api import L2Distance, CosineDistance, softmax_activation
from thinc.api import Ragged
from thinc import registry
from thinc.util import has_torch, to_categorical
from hypothesis import given, settings
from hypothesis.strategies import integers, floats
from thinc.legacy import loss


ALL_XP = [numpy]
try:
    import cupy

    ALL_XP.append(cupy)
except ImportError:
    pass


softmax_func = partial(softmax_activation(), is_train=False)
MAX_EXAMPLES = 50
# some simple arrays
scores0 = numpy.zeros((3, 3), dtype="f")
labels0 = numpy.asarray([0, 1, 1], dtype="i")

# a few more diverse ones to test realistic values
guesses1 = numpy.asarray(
    [[0.1, 0.5, 0.4], [0.4, 0.3, 0.3], [0, 1, 0], [0.1, 0.05, 0.85]], dtype="f"
)
guesses1_legacy = numpy.asarray(
    [[0.1, 0.5, 0.6], [0.4, 0.6, 0.3], [1, 1, 1], [0, 0, 0]], dtype="f"
)
labels1 = numpy.asarray([2, 1, 0, 2])
labels1_full = numpy.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype="f")
labels1_strings = ["C", "B", "A", "C"]
d_guesses1 = numpy.array(
    [
        [0.025, 0.125, -0.15],
        [0.1, -0.175, 0.075],
        [-0.25, 0.25, 0.0],
        [0.025, 0.0125, -0.0375],
    ],
    dtype="f",
)
d_guesses1_seq = numpy.array(
    [
        [0.05, 0.25, -0.3],
        [0.2, -0.35, 0.15],
        [-0.5, 0.5, 0.0],
        [0.05, 0.025, -0.075],
    ],
    dtype="f",
)
d_guesses1_0_missing = numpy.array(
    [
        [0.025, 0.125, -0.15],
        [0.1, -0.175, 0.075],
        [0.0, 0.0, 0.0],
        [0.025, 0.0125, -0.0375],
    ],
    dtype="f",
)
d_guesses1_sum = numpy.array(
    [
        [0.1, 0.5, -0.6],
        [0.4, -0.7, 0.3],
        [-1.0, 1.0, 0.0],
        [0.1, 0.05, -0.15],
    ],
    dtype="f",
)
loss1 = 5.75151207
loss1_seq = 11.50302410
loss1_0_missing = 0.57069561
guesses2 = numpy.asarray([[0.2, 0.3, 0.5]])
guesses2_legacy = numpy.asarray([[0.2, 0.3, 0.0]])
labels2 = numpy.asarray([1])
labels2_strings = ["B"]
d_guesses2_sum = numpy.asarray([[0.2, -0.7, 0.5]])
sequence_loss = 24.210021096627
eps = 1e-6


ce_factory = registry.get("losses", "CategoricalCrossentropy.v4")

sparse_ce_factory = registry.get("losses", "SparseCategoricalCrossentropy.v4")

seq_ce_factory = registry.get("losses", "SequenceCategoricalCrossentropy.v4")


def _get_legacy_cross_entropy(version: int, **kwargs):
    return registry.get("losses", f"CategoricalCrossentropy.v{version}")(**kwargs)


def _get_legacy_seq_cross_entropy(version: int, **kwargs):
    return registry.get("losses", f"SequenceCategoricalCrossentropy.v{version}")(
        **kwargs
    )


def test_cross_entropy_types_shapes():
    sparse_cross_entropy = ce_factory()
    cross_entropy = ce_factory()
    sparse_seq_cross_entropy = seq_ce_factory()
    seq_cross_entropy = seq_ce_factory(sparse=False)
    d_scores_sparse = sparse_cross_entropy.get_grad(guesses1, labels1_full)
    d_scores = cross_entropy.get_grad(guesses1, labels1_full)
    assert d_scores_sparse.dtype == "float32"
    assert d_scores.dtype == "float32"
    assert d_scores_sparse.shape == guesses1.shape
    assert d_scores.shape == guesses1.shape
    d_scores_sparse = sparse_seq_cross_entropy.get_grad([guesses1], [labels1])
    d_scores = seq_cross_entropy.get_grad([guesses1], [labels1_full])
    assert d_scores_sparse[0].dtype == "float32"
    assert d_scores[0].dtype == "float32"
    assert d_scores_sparse[0].shape == guesses1.shape
    assert d_scores[0].shape == guesses1.shape
    assert sparse_seq_cross_entropy.get_grad([], []) == []
    assert seq_cross_entropy.get_grad([], []) == []
    d_scores_ragged = cross_entropy.get_grad(
        Ragged(numpy.array(guesses1), lengths=[3, 1]), labels1_full
    )
    assert isinstance(d_scores_ragged, Ragged)
    assert d_scores_ragged.dataXd.dtype == "float32"
    assert d_scores_ragged.dataXd.shape == guesses1.shape


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_cross_entropy_types_shapes(version):
    cross_entropy = _get_legacy_cross_entropy(version)
    seq_cross_entropy = _get_legacy_seq_cross_entropy(version)
    d_scores = cross_entropy.get_grad(scores0, labels0)
    assert d_scores.dtype == "float32"
    assert d_scores.shape == scores0.shape
    d_scores = seq_cross_entropy.get_grad([scores0], [labels0])
    assert d_scores[0].dtype == "float32"
    assert d_scores[0].shape == scores0.shape
    assert seq_cross_entropy.get_grad([], []) == []


@pytest.mark.skipif(not has_torch, reason="needs PyTorch")
@pytest.mark.parametrize("xp", ALL_XP)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(
    n_samples=integers(min_value=1, max_value=100),
    n_classes=integers(min_value=1, max_value=100),
    low=floats(min_value=-20, max_value=10),
    offset=floats(min_value=1, max_value=10),
)
def test_compare_cross_entropy_to_torch(xp, n_samples, n_classes, low, offset):
    import torch

    sparse_loss_sum = sparse_ce_factory(normalize=False)
    sparse_loss_mean = sparse_ce_factory()
    loss_sum = ce_factory(normalize=False)
    loss_mean = ce_factory()
    torch_loss_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    torch_loss_mean = torch.nn.CrossEntropyLoss()
    logits = xp.random.uniform(low, low + offset, (n_samples, n_classes))
    labels = xp.random.randint(0, n_classes, n_samples)
    labels_full = to_categorical(labels, n_classes=n_classes)
    torch_logits = torch.tensor(logits, requires_grad=True)
    torch_labels = torch.tensor(labels, dtype=torch.long)
    probs, _ = softmax_func(logits)
    d_sum_sparse, l_sum_sparse = sparse_loss_sum(probs, labels)
    d_sum, l_sum = loss_sum(probs, labels_full)
    torch_l_sum = torch_loss_sum(torch_logits, torch_labels)
    torch_l_sum.backward()
    torch_d_sum = torch_logits.grad
    torch_logits = torch.tensor(logits, requires_grad=True)
    d_mean_sparse, l_mean_sparse = sparse_loss_mean(probs, labels)
    d_mean, l_mean = loss_mean(probs, labels_full)
    torch_l_mean = torch_loss_mean(torch_logits, torch_labels)
    torch_l_mean.backward()
    torch_d_mean = torch_logits.grad
    assert xp.isclose(float(l_sum), float(torch_l_sum), atol=1e-06)
    assert xp.allclose(d_sum, torch_d_sum.numpy())
    assert xp.isclose(float(l_mean), float(torch_l_mean))
    assert xp.allclose(d_mean, torch_d_mean.numpy())
    assert xp.isclose(float(l_sum_sparse), float(torch_l_sum), atol=1e-06)
    assert xp.allclose(d_sum_sparse, torch_d_sum.numpy())
    assert xp.isclose(float(l_mean_sparse), float(torch_l_mean))
    assert xp.allclose(d_mean_sparse, torch_d_mean.numpy())


@pytest.mark.parametrize("dist", [CosineDistance(ignore_zeros=True), L2Distance()])
@pytest.mark.parametrize("vect", [scores0, guesses1, guesses2])
def test_equal_distance(dist, vect):
    assert int(dist.get_grad(vect, vect)[0][0]) == pytest.approx(0, abs=eps)
    assert dist.get_loss(vect, vect) == pytest.approx(0, abs=eps)


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("vect", [scores0, guesses1_legacy, guesses2_legacy])
def test_equal_legacy_cross_entropy(vect, version):
    cross_entropy = _get_legacy_cross_entropy(version)
    assert int(cross_entropy.get_grad(vect, vect)[0][0]) == pytest.approx(0, abs=eps)
    assert cross_entropy.get_loss(vect, vect) == pytest.approx(0, abs=eps)
    assert cross_entropy.get_loss(vect, vect) == pytest.approx(0, abs=eps)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_cross_entropy_absent_labels(version):
    cross_entropy = _get_legacy_cross_entropy(version, names=["cat", "dog", "rat"])
    assert cross_entropy.get_loss(scores0, [None, None, None]) == pytest.approx(
        0, abs=eps
    )


@pytest.mark.parametrize(
    "guesses, labels, grad, grad_seq, loss, loss_seq",
    [
        (guesses1, labels1_full, d_guesses1, d_guesses1_seq, loss1, loss1_seq),
    ],
)
def test_categorical_crossentropy(guesses, labels, grad, grad_seq, loss, loss_seq):
    cross_entropy = ce_factory()
    d_scores = cross_entropy.get_grad(guesses, labels)
    loss_val = cross_entropy.get_loss(guesses, labels)
    assert d_scores.shape == guesses.shape
    assert numpy.allclose(d_scores, grad)
    assert numpy.isclose(loss_val, loss)

    # Test with Ragged inputs
    d_scores_ragged = cross_entropy.get_grad(Ragged(guesses, lengths=[3, 1]), labels)
    loss_ragged = cross_entropy.get_loss(Ragged(guesses, lengths=[3, 1]), labels)
    assert d_scores_ragged.dataXd.shape == guesses.shape
    assert numpy.allclose(d_scores_ragged.dataXd, grad_seq)
    assert numpy.isclose(loss_ragged, loss_seq)


@pytest.mark.parametrize(
    "guesses, labels, grad, grad_seq, loss, loss_seq",
    [
        (guesses1, labels1, d_guesses1, d_guesses1_seq, loss1, loss1_seq),
    ],
)
def test_sparse_categorical_crossentropy(
    guesses, labels, grad, grad_seq, loss, loss_seq
):
    cross_entropy = sparse_ce_factory()
    d_scores = cross_entropy.get_grad(guesses, labels)
    loss_val = cross_entropy.get_loss(guesses, labels)
    assert d_scores.shape == guesses.shape
    assert numpy.allclose(d_scores, grad)
    assert numpy.isclose(loss_val, loss)

    # Test with Ragged inputs
    d_scores_ragged = cross_entropy.get_grad(Ragged(guesses, lengths=[3, 1]), labels)
    loss_ragged = cross_entropy.get_loss(Ragged(guesses, lengths=[3, 1]), labels)
    assert d_scores_ragged.dataXd.shape == guesses.shape
    assert numpy.allclose(d_scores_ragged.dataXd, grad_seq)
    assert numpy.isclose(loss_ragged, loss_seq)


@pytest.mark.parametrize(
    "guesses, labels", [(guesses1_legacy, labels1), (guesses1_legacy, labels1_full)]
)
@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_categorical_crossentropy(guesses, labels, version):
    cross_entropy_normalize = _get_legacy_cross_entropy(version, normalize=True)
    d_scores = cross_entropy_normalize.get_grad(guesses, labels)
    assert d_scores.shape == guesses.shape

    # The normalization divides the difference (e.g. 0.4) by the number of vectors (4)
    assert d_scores[1][0] == pytest.approx(0.1, abs=eps)
    assert d_scores[1][1] == pytest.approx(-0.1, abs=eps)

    # The third vector predicted all labels, but only the first one was correct
    assert d_scores[2][0] == pytest.approx(0, abs=eps)
    assert d_scores[2][1] == pytest.approx(0.25, abs=eps)
    assert d_scores[2][2] == pytest.approx(0.25, abs=eps)

    # The fourth vector predicted no labels but should have predicted the last one
    assert d_scores[3][0] == pytest.approx(0, abs=eps)
    assert d_scores[3][1] == pytest.approx(0, abs=eps)
    assert d_scores[3][2] == pytest.approx(-0.25, abs=eps)

    loss = cross_entropy_normalize.get_loss(guesses, labels)
    assert loss == pytest.approx(0.239375, abs=eps)


def test_crossentropy_incorrect_scores_targets():
    labels = numpy.asarray([2])
    labels_full = numpy.asarray([[0.0, 0.0, 1.0]])
    cross_entropy = ce_factory()
    sparse_cross_entropy = sparse_ce_factory()

    guesses_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        cross_entropy.get_grad(guesses_neg, labels_full)
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        sparse_cross_entropy.get_grad(guesses_neg, labels)

    guesses_dont_sum_one = numpy.asarray([[0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        cross_entropy.get_grad(guesses_dont_sum_one, labels_full)
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        sparse_cross_entropy.get_grad(guesses_dont_sum_one, labels)

    guesses_larger_than_one = numpy.asarray([[1.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        cross_entropy.get_grad(guesses_larger_than_one, labels_full)
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        sparse_cross_entropy.get_grad(guesses_larger_than_one, labels)

    guesses_ok = numpy.asarray([[0.1, 0.4, 0.5]])
    targets_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        cross_entropy.get_grad(guesses_ok, targets_neg)

    targets_larger_than_one = numpy.asarray([[2.0, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        cross_entropy.get_grad(guesses_ok, targets_larger_than_one)

    targets_dont_sum_one = numpy.asarray([[0.9, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        cross_entropy.get_grad(guesses_ok, targets_dont_sum_one)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_categorical_cross_entropy_incorrect_scores_targets(version):
    labels = numpy.asarray([2])
    cross_entropy_normalize = _get_legacy_cross_entropy(version, normalize=True)
    guesses_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        cross_entropy_normalize.get_grad(guesses_neg, labels)

    guesses_larger_than_one = numpy.asarray([[1.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        cross_entropy_normalize.get_grad(guesses_larger_than_one, labels)

    guesses_ok = numpy.asarray([[0.1, 0.4, 0.5]])
    targets_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        cross_entropy_normalize.get_grad(guesses_ok, targets_neg)

    targets_larger_than_one = numpy.asarray([[2.0, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        cross_entropy_normalize.get_grad(guesses_ok, targets_larger_than_one)


@pytest.mark.parametrize(
    "guesses, labels, grad, missing_value",
    [
        (guesses1, [2, 1, 0, 2], d_guesses1_0_missing, 0),
        (guesses1, labels1, d_guesses1_0_missing, 0),
        (guesses1, labels1_strings, d_guesses1_0_missing, "A"),
    ],
)
def test_sparse_crossentropy_missing(guesses, labels, grad, missing_value):
    if missing_value == "A":
        names = ["A", "B", "C"]
    else:
        names = None
    sparse_cross_entropy = sparse_ce_factory(missing_value=missing_value, names=names)
    d_scores = sparse_cross_entropy.get_grad(guesses, labels)
    assert d_scores.shape == guesses.shape
    assert numpy.allclose(d_scores, grad)
    loss = sparse_cross_entropy.get_loss(guesses, labels)
    assert numpy.isclose(loss, loss1_0_missing)


@pytest.mark.parametrize(
    "guesses, labels",
    [(guesses1_legacy, [2, 1, 0, 2])],
)
@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_categorical_crossentropy_int_list_missing(guesses, labels, version):
    cross_entropy_normalize_missing = _get_legacy_cross_entropy(
        version, normalize=True, missing_value=0
    )
    d_scores = cross_entropy_normalize_missing.get_grad(guesses, labels)
    assert d_scores.shape == guesses.shape

    # The normalization divides the difference (e.g. 0.4) by the number of vectors (4)
    assert d_scores[1][0] == pytest.approx(0.1, abs=eps)
    assert d_scores[1][1] == pytest.approx(-0.1, abs=eps)

    # Label 0 is masked, because it represents the missing value
    assert d_scores[2][0] == 0.0
    assert d_scores[2][1] == 0.0
    assert d_scores[2][2] == 0.0

    # The fourth vector predicted no labels but should have predicted the last one
    assert d_scores[3][0] == pytest.approx(0, abs=eps)
    assert d_scores[3][1] == pytest.approx(0, abs=eps)
    assert d_scores[3][2] == pytest.approx(-0.25, abs=eps)

    loss = cross_entropy_normalize_missing.get_loss(guesses, labels)
    assert loss == pytest.approx(0.114375, abs=eps)


@pytest.mark.parametrize(
    "guesses, labels, grad",
    [
        (guesses1, labels1_full, d_guesses1_0_missing),
    ],
)
def test_categorical_crossentropy_missing(guesses, labels, grad):
    cross_entropy = ce_factory(missing_value=0)
    d_scores = cross_entropy.get_grad(guesses, labels)
    assert d_scores.shape == guesses.shape
    assert numpy.allclose(d_scores, grad)

    loss = CategoricalCrossentropy(normalize=True, missing_value=0).get_loss(
        guesses, labels
    )
    assert numpy.isclose(loss, loss1_0_missing)


@pytest.mark.parametrize(
    "guesses, labels", [(guesses1_legacy, labels1), (guesses1_legacy, labels1_full)]
)
@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_categorical_crossentropy_missing(guesses, labels, version):
    cross_entropy_normalize_missing = _get_legacy_cross_entropy(
        version, normalize=True, missing_value=0
    )
    d_scores = cross_entropy_normalize_missing.get_grad(guesses, labels)
    assert d_scores.shape == guesses.shape

    # The normalization divides the difference (e.g. 0.4) by the number of vectors (4)
    assert d_scores[1][0] == pytest.approx(0.1, abs=eps)
    assert d_scores[1][1] == pytest.approx(-0.1, abs=eps)

    # Label 0 is masked, because it represents the missing value
    assert d_scores[2][0] == 0.0
    assert d_scores[2][1] == 0.0
    assert d_scores[2][2] == 0.0

    # The fourth vector predicted no labels but should have predicted the last one
    assert d_scores[3][0] == pytest.approx(0, abs=eps)
    assert d_scores[3][1] == pytest.approx(0, abs=eps)
    assert d_scores[3][2] == pytest.approx(-0.25, abs=eps)

    loss = cross_entropy_normalize_missing.get_loss(guesses, labels)
    assert loss == pytest.approx(0.114375, abs=eps)


@pytest.mark.parametrize(
    "guesses, labels, names, grad, loss",
    [
        (
            [guesses1, guesses2],
            [labels1, labels2],
            [],
            [d_guesses1_sum, d_guesses2_sum],
            sequence_loss,
        ),
        (
            [guesses1, guesses2],
            [labels1_strings, labels2_strings],
            ["A", "B", "C"],
            [d_guesses1_sum, d_guesses2_sum],
            sequence_loss,
        ),
    ],
)
def test_sequence_sparse_crossentropy(guesses, labels, names, grad, loss):
    sparse_seq_cross_entropy_sum = seq_ce_factory(names=names, normalize=False)
    sparse_seq_cross_entropy = seq_ce_factory(names=names, normalize=True)
    d_scores = sparse_seq_cross_entropy_sum.get_grad(guesses, labels)
    assert numpy.allclose(d_scores[0], grad[0])
    assert numpy.allclose(d_scores[1], grad[1])
    # The normalization divides the difference (e.g. 0.4) by the number of seqs
    d_scores = sparse_seq_cross_entropy.get_grad(guesses, labels)
    assert numpy.allclose(d_scores[0], grad[0] / 2.0)
    assert numpy.allclose(d_scores[1], grad[1] / 2.0)
    loss_val = sparse_seq_cross_entropy.get_loss(guesses, labels)
    assert numpy.isclose(loss_val, loss)
    d_scores, loss_val = sparse_seq_cross_entropy_sum(guesses, labels)
    assert numpy.isclose(loss_val, loss)
    assert numpy.allclose(d_scores[0], grad[0])
    assert numpy.allclose(d_scores[1], grad[1])


@pytest.mark.parametrize(
    "guesses, labels, grad, loss",
    [([guesses1], [labels1_full], [d_guesses1_sum], [23.00604829563447])],
)
def test_sequence_crossentropy(guesses, labels, grad, loss):
    seq_cross_entropy = seq_ce_factory(sparse=False, normalize=False)
    d_scores = seq_cross_entropy.get_grad(guesses, labels)
    assert numpy.allclose(d_scores[0], grad[0])
    # The normalization divides the difference (e.g. 0.4) by the number of seqs
    loss_val = seq_cross_entropy.get_loss(guesses, labels)
    assert numpy.isclose(loss_val, loss)
    d_scores, loss_val = seq_cross_entropy(guesses, labels)
    assert numpy.isclose(loss_val, loss)
    assert numpy.allclose(d_scores[0], grad[0])


@pytest.mark.parametrize(
    "guesses, labels, names",
    [
        ([guesses1_legacy, guesses2_legacy], [labels1, labels2], []),
        ([guesses1_legacy, guesses2_legacy], [labels1_full, labels2], []),
        (
            [guesses1_legacy, guesses2_legacy],
            [labels1_strings, labels2_strings],
            ["A", "B", "C"],
        ),
    ],
)
@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_sequence_categorical_crossentropy(guesses, labels, names, version):
    seq_cross_entropy_names = _get_legacy_seq_cross_entropy(
        version, normalize=False, names=names
    )
    seq_cross_entropy_names_normalize = _get_legacy_seq_cross_entropy(
        version, normalize=True, names=names
    )
    d_scores = seq_cross_entropy_names.get_grad(guesses, labels)
    d_scores1 = d_scores[0]
    d_scores2 = d_scores[1]
    assert d_scores1.shape == guesses1.shape
    assert d_scores2.shape == guesses2.shape
    assert d_scores1[1][0] == pytest.approx(0.4, abs=eps)
    assert d_scores1[1][1] == pytest.approx(-0.4, abs=eps)
    # The normalization divides the difference (e.g. 0.4) by the number of seqs
    d_scores = seq_cross_entropy_names_normalize.get_grad(guesses, labels)
    d_scores1 = d_scores[0]
    d_scores2 = d_scores[1]

    assert d_scores1[1][0] == pytest.approx(0.2, abs=eps)
    assert d_scores1[1][1] == pytest.approx(-0.2, abs=eps)

    # The third vector predicted all labels, but only the first one was correct
    assert d_scores1[2][0] == pytest.approx(0, abs=eps)
    assert d_scores1[2][1] == pytest.approx(0.5, abs=eps)
    assert d_scores1[2][2] == pytest.approx(0.5, abs=eps)

    # The fourth vector predicted no labels but should have predicted the last one
    assert d_scores1[3][0] == pytest.approx(0, abs=eps)
    assert d_scores1[3][1] == pytest.approx(0, abs=eps)
    assert d_scores1[3][2] == pytest.approx(-0.5, abs=eps)

    # Test the second batch
    assert d_scores2[0][0] == pytest.approx(0.1, abs=eps)
    assert d_scores2[0][1] == pytest.approx(-0.35, abs=eps)

    loss = seq_cross_entropy_names_normalize.get_loss(guesses, labels)
    assert loss == pytest.approx(1.09, abs=eps)


@pytest.mark.parametrize(
    "guesses, labels, names, grad",
    [
        (
            [guesses1],
            [["A", "!A", "", "!C"]],
            ["A", "B", "C"],
            numpy.array(
                [
                    [-0.9, 0.5, 0.4],  # First is correct
                    [0.4, 0.0, 0.0],  # Not first one
                    [0.0, 0.0, 0.0],  # Missing
                    [0.0, 0.0, 0.85],  # Not last one
                ]
            ),
        )
    ],
)
def test_sequence_crossentropy_missing_negative(guesses, labels, names, grad):
    sparse_seq_ce = seq_ce_factory(
        names=names, normalize=False, neg_prefix="!", missing_value=""
    )
    d_scores = sparse_seq_ce.get_grad(guesses, labels)
    assert numpy.allclose(d_scores, grad)


@pytest.mark.parametrize(
    "guesses, labels, names",
    [
        ([guesses1_legacy], [["A", "!A", "", "!C"]], ["A", "B", "C"]),
    ],
)
@pytest.mark.parametrize("version", [3])
def test_legacy_sequence_categorical_missing_negative(guesses, labels, names, version):
    seq_cross_entropy = _get_legacy_seq_cross_entropy(
        version, normalize=False, names=names, neg_prefix="!", missing_value=""
    )
    d_scores = seq_cross_entropy.get_grad(guesses, labels)
    d_scores0 = d_scores[0]

    # [0.1, 0.5, 0.6] should be A
    assert d_scores0[0][0] == pytest.approx(-0.9, abs=eps)
    assert d_scores0[0][1] == pytest.approx(0.5, abs=eps)
    assert d_scores0[0][2] == pytest.approx(0.6, abs=eps)

    # [0.4, 0.6, 0.3] should NOT be A
    assert d_scores0[1][0] == pytest.approx(0.4, abs=eps)
    assert d_scores0[1][1] == pytest.approx(0.0, abs=eps)
    assert d_scores0[1][2] == pytest.approx(0.0, abs=eps)

    # [1, 1, 1] has missing gold label
    assert d_scores0[2][0] == pytest.approx(0.0, abs=eps)
    assert d_scores0[2][1] == pytest.approx(0.0, abs=eps)
    assert d_scores0[2][2] == pytest.approx(0.0, abs=eps)

    # [0.0, 0.0, 0.0] should NOT be C
    assert d_scores0[3][0] == pytest.approx(0.0, abs=eps)
    assert d_scores0[3][1] == pytest.approx(0.0, abs=eps)
    assert d_scores0[3][2] == pytest.approx(0.0, abs=eps)


def test_L2():
    # L2 loss = 2²+4²=20 (or normalized: 1²+2²=5)
    vec1 = numpy.asarray([[1, 2], [8, 9]])
    vec2 = numpy.asarray([[1, 2], [10, 5]])
    d_vecs = L2Distance().get_grad(vec1, vec2)
    assert d_vecs.shape == vec1.shape
    numpy.testing.assert_allclose(
        d_vecs[0], numpy.zeros(d_vecs[0].shape), rtol=eps, atol=eps
    )

    loss_not_normalized = L2Distance(normalize=False).get_loss(vec1, vec2)
    assert loss_not_normalized == pytest.approx(20, abs=eps)

    loss_normalized = L2Distance(normalize=True).get_loss(vec1, vec2)
    assert loss_normalized == pytest.approx(5, abs=eps)


def test_cosine_orthogonal():
    # These are orthogonal, i.e. loss is 1
    vec1 = numpy.asarray([[0, 2], [0, 5]])
    vec2 = numpy.asarray([[8, 0], [7, 0]])

    d_vecs = CosineDistance(normalize=True).get_grad(vec1, vec2)
    assert d_vecs.shape == vec1.shape
    assert d_vecs[0][0] < 0
    assert d_vecs[0][1] > 0
    assert d_vecs[1][0] < 0
    assert d_vecs[1][1] > 0

    loss_not_normalized = CosineDistance(normalize=False).get_loss(vec1, vec2)
    assert loss_not_normalized == pytest.approx(2, abs=eps)

    loss_normalized = CosineDistance(normalize=True).get_loss(vec1, vec2)
    assert loss_normalized == pytest.approx(1, abs=eps)


def test_cosine_equal():
    # These 3 vectors are equal when measured with Cosine similarity, i.e. loss is 0
    vec1 = numpy.asarray([[1, 2], [8, 9], [3, 3]])
    vec2 = numpy.asarray([[1, 2], [80, 90], [300, 300]])

    d_vec1 = CosineDistance().get_grad(vec1, vec2)
    assert d_vec1.shape == vec1.shape
    numpy.testing.assert_allclose(d_vec1, numpy.zeros(d_vec1.shape), rtol=eps, atol=eps)

    loss_not_normalized = CosineDistance(normalize=False).get_loss(vec1, vec2)
    assert loss_not_normalized == pytest.approx(0, abs=eps)

    loss_normalized = CosineDistance(normalize=True).get_loss(vec1, vec2)
    assert loss_normalized == pytest.approx(0, abs=eps)


def test_cosine_unmatched():
    vec1 = numpy.asarray([[1, 2, 3]])
    vec2 = numpy.asarray([[1, 2]])
    with pytest.raises(ValueError):
        CosineDistance().get_grad(vec1, vec2)


@pytest.mark.parametrize(
    "name,kwargs,args",
    [
        ("CategoricalCrossentropy.v1", {}, (guesses1, labels1)),
        ("SequenceCategoricalCrossentropy.v1", {}, ([guesses1], [labels1])),
        ("CategoricalCrossentropy.v2", {"neg_prefix": "!"}, (guesses1, labels1)),
        ("CategoricalCrossentropy.v3", {"neg_prefix": "!"}, (guesses1, labels1)),
        ("SparseCategoricalCrossentropy.v4", {"neg_prefix": "!"}, (guesses1, labels1)),
        ("CategoricalCrossentropy.v4", {}, (guesses1, labels1_full)),
        (
            "SequenceCategoricalCrossentropy.v2",
            {"neg_prefix": "!"},
            ([guesses1], [labels1]),
        ),
        (
            "SequenceCategoricalCrossentropy.v3",
            {"neg_prefix": "!"},
            ([guesses1], [labels1]),
        ),
        (
            "SequenceCategoricalCrossentropy.v4",
            {"neg_prefix": "!"},
            ([guesses1], [labels1]),
        ),
        ("L2Distance.v1", {}, (scores0, scores0)),
        (
            "CosineDistance.v1",
            {"normalize": True, "ignore_zeros": True},
            (scores0, scores0),
        ),
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
