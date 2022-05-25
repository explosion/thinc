import pytest
import numpy
from functools import partial
from thinc.api import CategoricalCrossentropy, SequenceCategoricalCrossentropy
from thinc.api import L2Distance, CosineDistance, softmax_activation
from thinc import registry
from thinc.util import has_torch
from hypothesis import given, settings
from hypothesis.strategies import integers, floats


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
    [[0.1, 0.5, 0.4], [0.4, 0.3, 0.3], [0, 1, 0], [0.1, 0.05, 0.85]]
)
labels1 = numpy.asarray([2, 1, 0, 2])
labels1_full = numpy.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
labels1_strings = ["C", "B", "A", "C"]
d_guesses1 = numpy.array(
    [
        [0.025, 0.125, -0.15],
        [0.1, -0.175, 0.075],
        [-0.25, 0.25, 0.0],
        [0.025, 0.0125, -0.0375],
    ]
)
d_guesses1_0_missing = numpy.array(
    [
        [0.025, 0.125, -0.15],
        [0.1, -0.175, 0.075],
        [0.0, 0.0, 0.0],
        [0.025, 0.0125, -0.0375],
    ]
)
d_guesses1_sum = numpy.array(
    [
        [0.1, 0.5, -0.6],
        [0.4, -0.7, 0.3],
        [-1.0, 1.0, 0.0],
        [0.1, 0.05, -0.15],
    ]
)
loss1 = 5.75151207
loss1_0_missing = 0.57069561
guesses2 = numpy.asarray([[0.2, 0.3, 0.5]])
labels2 = numpy.asarray([1])
labels2_strings = ["B"]
d_guesses2_sum = numpy.asarray([[0.2, -0.7, 0.5]])
sequence_loss = 24.210021096627
eps = 0.0001


def test_cross_entropy_types_shapes():
    d_scores = CategoricalCrossentropy().get_grad(guesses1, labels1)
    assert d_scores.dtype == "float32"
    assert d_scores.shape == guesses1.shape
    d_scores = SequenceCategoricalCrossentropy().get_grad([guesses1], [labels1])
    assert d_scores[0].dtype == "float32"
    assert d_scores[0].shape == guesses1.shape
    assert SequenceCategoricalCrossentropy().get_grad([], []) == []


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

    loss_sum = CategoricalCrossentropy(normalize=False)
    loss_mean = CategoricalCrossentropy()
    torch_loss_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    torch_loss_mean = torch.nn.CrossEntropyLoss()
    logits = xp.random.uniform(low, low + offset, (n_samples, n_classes))
    labels = xp.random.randint(0, n_classes, n_samples)
    torch_logits = torch.tensor(logits, requires_grad=True)
    torch_labels = torch.tensor(labels, dtype=torch.long)
    probs, _ = softmax_func(logits)
    d_sum, l_sum = loss_sum(probs, labels)
    torch_l_sum = torch_loss_sum(torch_logits, torch_labels)
    torch_l_sum.backward()
    torch_d_sum = torch_logits.grad
    torch_logits = torch.tensor(logits, requires_grad=True)
    d_mean, l_mean = loss_mean(probs, labels)
    torch_l_mean = torch_loss_mean(torch_logits, torch_labels)
    torch_l_mean.backward()
    torch_d_mean = torch_logits.grad
    assert xp.isclose(float(l_sum), float(torch_l_sum), atol=1e-06)
    assert xp.allclose(d_sum, torch_d_sum.numpy())
    assert xp.isclose(float(l_mean), float(torch_l_mean))
    assert xp.allclose(d_mean, torch_d_mean.numpy())


@pytest.mark.parametrize("dist", [CosineDistance(ignore_zeros=True), L2Distance()])
@pytest.mark.parametrize("vect", [scores0, guesses1, guesses2])
def test_equal_distance(dist, vect):
    assert int(dist.get_grad(vect, vect)[0][0]) == pytest.approx(0, eps)
    assert dist.get_loss(vect, vect) == pytest.approx(0, eps)


@pytest.mark.parametrize(
    "guesses, labels, grad, loss",
    [
        (guesses1, labels1, d_guesses1, loss1),
        (guesses1, labels1_full, d_guesses1, loss1),
    ],
)
def test_categorical_crossentropy(guesses, labels, grad, loss):
    d_scores = CategoricalCrossentropy(normalize=True).get_grad(guesses, labels)
    loss_val = CategoricalCrossentropy(normalize=True).get_loss(guesses, labels)
    assert d_scores.shape == guesses.shape
    assert numpy.allclose(d_scores, grad)
    assert numpy.isclose(loss_val, loss)


def test_crossentropy_incorrect_scores_targets():
    labels = numpy.asarray([2])

    guesses_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        CategoricalCrossentropy(normalize=True).get_grad(guesses_neg, labels)

    guesses_dont_sum_one = numpy.asarray([[0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        CategoricalCrossentropy(normalize=True).get_grad(guesses_dont_sum_one, labels)

    guesses_larger_than_one = numpy.asarray([[1.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*guesses"):
        CategoricalCrossentropy(normalize=True).get_grad(
            guesses_larger_than_one, labels
        )

    guesses_ok = numpy.asarray([[0.1, 0.4, 0.5]])
    targets_neg = numpy.asarray([[-0.1, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        CategoricalCrossentropy(normalize=True).get_grad(guesses_ok, targets_neg)

    targets_larger_than_one = numpy.asarray([[2.0, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        CategoricalCrossentropy(normalize=True).get_grad(
            guesses_ok, targets_larger_than_one
        )

    targets_dont_sum_one = numpy.asarray([[0.9, 0.5, 0.6]])
    with pytest.raises(ValueError, match=r"Cannot calculate.*truth"):
        CategoricalCrossentropy(normalize=True).get_grad(
            guesses_ok, targets_dont_sum_one
        )


@pytest.mark.parametrize(
    "guesses, labels, grad",
    [(guesses1, [2, 1, 0, 2], d_guesses1_0_missing)],
)
def test_categorical_crossentropy_int_list_missing(guesses, labels, grad):
    d_scores = CategoricalCrossentropy(normalize=True, missing_value=0).get_grad(
        guesses, labels
    )
    assert d_scores.shape == guesses.shape
    assert numpy.allclose(d_scores, grad)

    loss = CategoricalCrossentropy(normalize=True, missing_value=0).get_loss(
        guesses, labels
    )
    assert numpy.isclose(loss, loss1_0_missing)


@pytest.mark.parametrize(
    "guesses, labels, grad",
    [
        (guesses1, labels1, d_guesses1_0_missing),
        (guesses1, labels1_full, d_guesses1_0_missing),
    ],
)
def test_categorical_crossentropy_missing(guesses, labels, grad):
    d_scores = CategoricalCrossentropy(normalize=True, missing_value=0).get_grad(
        guesses, labels
    )
    assert d_scores.shape == guesses.shape
    assert numpy.allclose(d_scores, grad)

    loss = CategoricalCrossentropy(normalize=True, missing_value=0).get_loss(
        guesses, labels
    )
    assert numpy.isclose(loss, loss1_0_missing)


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
            [labels1_full, labels2],
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
def test_sequence_categorical_crossentropy(guesses, labels, names, grad, loss):
    d_scores = SequenceCategoricalCrossentropy(normalize=False, names=names).get_grad(
        guesses, labels
    )
    assert numpy.allclose(d_scores[0], grad[0])
    assert numpy.allclose(d_scores[1], grad[1])
    # The normalization divides the difference (e.g. 0.4) by the number of seqs
    d_scores = SequenceCategoricalCrossentropy(normalize=True, names=names).get_grad(
        guesses, labels
    )
    assert numpy.allclose(d_scores[0], grad[0] / 2.0)
    assert numpy.allclose(d_scores[1], grad[1] / 2.0)
    loss_val = SequenceCategoricalCrossentropy(normalize=True, names=names).get_loss(
        guesses, labels
    )
    assert numpy.isclose(loss_val, loss)
    loss_func = SequenceCategoricalCrossentropy(normalize=False, names=names)
    d_scores, loss_val = loss_func(guesses, labels)
    assert numpy.isclose(loss_val, loss)
    assert numpy.allclose(d_scores[0], grad[0])
    assert numpy.allclose(d_scores[1], grad[1])

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
def test_sequence_categorical_missing_negative(guesses, labels, names, grad):
    d_scores = SequenceCategoricalCrossentropy(
        normalize=False, names=names, neg_prefix="!", missing_value=""
    ).get_grad(guesses, labels)
    assert numpy.allclose(d_scores, grad)


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
    assert loss_not_normalized == pytest.approx(20, eps)

    loss_normalized = L2Distance(normalize=True).get_loss(vec1, vec2)
    assert loss_normalized == pytest.approx(5, eps)


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
    assert loss_not_normalized == pytest.approx(2, eps)

    loss_normalized = CosineDistance(normalize=True).get_loss(vec1, vec2)
    assert loss_normalized == pytest.approx(1, eps)


def test_cosine_equal():
    # These 3 vectors are equal when measured with Cosine similarity, i.e. loss is 0
    vec1 = numpy.asarray([[1, 2], [8, 9], [3, 3]])
    vec2 = numpy.asarray([[1, 2], [80, 90], [300, 300]])

    d_vec1 = CosineDistance().get_grad(vec1, vec2)
    assert d_vec1.shape == vec1.shape
    numpy.testing.assert_allclose(d_vec1, numpy.zeros(d_vec1.shape), rtol=eps, atol=eps)

    loss_not_normalized = CosineDistance(normalize=False).get_loss(vec1, vec2)
    assert loss_not_normalized == pytest.approx(0, eps)

    loss_normalized = CosineDistance(normalize=True).get_loss(vec1, vec2)
    assert loss_normalized == pytest.approx(0, eps)


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
        ("CategoricalCrossentropy.v4", {"neg_prefix": "!"}, (guesses1, labels1)),
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
