import pytest
import numpy
from thinc.api import categorical_crossentropy, L1_distance, cosine_distance
from thinc import registry


@pytest.mark.parametrize("shape,labels", [([4, 3], [0, 2, 2, 2])])
def test_loss(shape, labels):
    scores = numpy.zeros(shape, dtype="f")
    labels = numpy.asarray(labels, dtype="i")
    d_scores = categorical_crossentropy(scores, labels)
    assert d_scores.dtype == "float32"
    assert d_scores.shape == scores.shape


def test_L1_distance():
    vec1 = numpy.asarray([[2]])
    vec2 = numpy.asarray([[3]])
    labels = [-1, -2, -3]
    d_vecs = L1_distance(vec1, vec2, labels)
    assert len(d_vecs) == 2


def test_cosine_distance():
    vec1 = numpy.asarray([[1, 2, 3]])
    vec2 = numpy.asarray([[1, 2, 4]])
    d_vec1 = cosine_distance(vec1, vec2)
    assert d_vec1.shape == vec1.shape


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("categorical_crossentropy.v0", {}),
        ("L1_distance.v0", {"margin": 0.5}),
        ("cosine_distance.v0", {"ignore_zeros": True}),
    ],
)
def test_loss_from_config(name, kwargs):
    """Test that losses are loaded and configured correctly from registry
    (as partials)."""
    cfg = {"test": {"@losses": name, **kwargs}}
    registry.make_from_config(cfg)["test"]
