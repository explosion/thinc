import pytest
import numpy
from thinc.api import categorical_crossentropy, L1_distance, cosine_distance


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
