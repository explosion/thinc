import pytest
import numpy
from thinc.api import CategoricalCrossentropy, SequenceCategoricalCrossentropy
from thinc.api import L2Distance, CosineDistance
from thinc import registry


scores = numpy.zeros((3, 3), dtype="f")
labels = numpy.asarray([0, 2, 2], dtype="i")

eps = 0.0001


def test_loss():
    d_scores = CategoricalCrossentropy().get_grad(scores, labels)
    assert d_scores.dtype == "float32"
    assert d_scores.shape == scores.shape
    d_scores = SequenceCategoricalCrossentropy().get_grad([scores], [labels])
    assert d_scores[0].dtype == "float32"
    assert d_scores[0].shape == scores.shape
    assert SequenceCategoricalCrossentropy().get_grad([], []) == []


def test_L2_distance():
    # L2 loss = 2²+4²=20
    vec1 = numpy.asarray([[1, 2], [8, 9]])
    vec2 = numpy.asarray([[1, 2], [10, 5]])
    d_vecs = L2Distance().get_grad(vec1, vec2)
    assert d_vecs.shape == vec1.shape
    numpy.testing.assert_allclose(d_vecs[0], numpy.zeros(d_vecs[0].shape), rtol=eps, atol=eps)
    loss = L2Distance().get_loss(vec1, vec2)
    assert loss == pytest.approx(20, eps)


def test_cosine_distance_orthogonal():
    # These are orthogonal, i.e. loss is 1
    vec1 = numpy.asarray([[0, 2]])
    vec2 = numpy.asarray([[8, 0]])

    d_vecs = CosineDistance().get_grad(vec1, vec2)
    assert d_vecs.shape == vec1.shape
    assert d_vecs[0][0] < 0
    assert d_vecs[0][1] > 0

    loss = CosineDistance().get_loss(vec1, vec2)
    assert loss == pytest.approx(1, eps)


def test_cosine_distance_equal():
    # These 3 vectors are equal when measured with Cosine similarity, i.e. loss is 0
    vec1 = numpy.asarray([[1, 2], [8, 9], [3, 3]])
    vec2 = numpy.asarray([[1, 2], [80, 90], [300, 300]])

    d_vec1 = CosineDistance().get_grad(vec1, vec2)
    assert d_vec1.shape == vec1.shape
    numpy.testing.assert_allclose(d_vec1, numpy.zeros(d_vec1.shape), rtol=eps, atol=eps)

    loss = CosineDistance().get_loss(vec1, vec2)
    assert loss == pytest.approx(0, eps)


def test_cosine_distance_unmatched():
    vec1 = numpy.asarray([[1, 2, 3]])
    vec2 = numpy.asarray([[1, 2]])
    with pytest.raises(ValueError):
        CosineDistance().get_grad(vec1, vec2)


@pytest.mark.parametrize(
    "name,kwargs,args",
    [
        #("CategoricalCrossentropy.v0", {}, (scores, labels)),
        #("SequenceCategoricalCrossentropy.v0", {}, ([scores], [labels])),
        ("L2Distance.v0", {}, (scores, scores)),
        ("CosineDistance.v0", {"ignore_zeros": True}, (scores, scores)),
    ],
)
def test_loss_from_config(name, kwargs, args):
    """Test that losses are loaded and configured correctly from registry
    (as partials)."""
    cfg = {"test": {"@losses": name, **kwargs}}
    func = registry.make_from_config(cfg)["test"]
    loss = func.get_grad(*args)
    if isinstance(loss, (list, tuple)):
        loss = loss[0]
    assert loss.ndim == 2
    func.get_loss(*args)
    func(*args)
