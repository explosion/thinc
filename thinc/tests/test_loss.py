import pytest
import numpy
from thinc.api import categorical_crossentropy, sequence_categorical_crossentropy
from thinc.api import L1_distance, cosine_distance
from thinc import registry


scores = numpy.zeros((3, 3), dtype="f")
labels = numpy.asarray([0, 2, 2], dtype="i")


def test_loss():
    d_scores = categorical_crossentropy(scores, labels)
    assert d_scores.dtype == "float32"
    assert d_scores.shape == scores.shape
    d_scores = sequence_categorical_crossentropy([scores], [labels])
    assert d_scores[0].dtype == "float32"
    assert d_scores[0].shape == scores.shape
    assert sequence_categorical_crossentropy([], []) == []


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
    "name,kwargs,args",
    [
        ("categorical_crossentropy.v0", {}, (scores, labels)),
        ("sequence_categorical_crossentropy.v0", {}, ([scores], [labels])),
        ("L1_distance.v0", {"margin": 0.5}, (scores, scores, labels)),
        ("cosine_distance.v0", {"ignore_zeros": True}, (scores, scores)),
    ],
)
def test_loss_from_config(name, kwargs, args):
    """Test that losses are loaded and configured correctly from registry
    (as partials)."""
    cfg = {"test": {"@losses": name, **kwargs}}
    func = registry.make_from_config(cfg)["test"]
    loss = func(*args)
    if isinstance(loss, (list, tuple)):
        loss = loss[0]
    assert loss.ndim == 2
