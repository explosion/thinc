import pytest
from mock import MagicMock
import numpy
from thinc.api import categorical_crossentropy, L1_distance, cosine_distance


@pytest.mark.parametrize("shape,labels", [([100, 100, 100], [-1, -1, -1])])
def test_loss(shape, labels):
    scores = MagicMock(spec=numpy.ndarray, shape=shape)
    loss = categorical_crossentropy(scores, labels)
    assert len(loss) == 2


def test_L1_distance():
    vec1 = numpy.asarray([[2]])
    vec2 = numpy.asarray([[3]])
    labels = [-1, -2, -3]
    loss = L1_distance(vec1, vec2, labels)
    assert len(loss) == 3
    assert numpy.array_equal(numpy.asarray([2, 3, 4]), loss[2])


def test_cosine_distance():
    vec1 = numpy.asarray([[1, 2, 3]])
    vec2 = numpy.asarray([[1, 2, 4]])
    loss = cosine_distance(vec1, vec2)
    assert len(loss) == 2
    vec2 = numpy.asarray([[0, 0, 0]])
    cosine_distance(vec1, vec2, ignore_zeros=True)
