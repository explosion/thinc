import pytest
from mock import MagicMock
from numpy import ndarray
from thinc.api import categorical_crossentropy


@pytest.mark.parametrize("shape,labels", [([100, 100, 100], [-1, -1, -1])])
def test_loss(shape, labels):
    scores = MagicMock(spec=ndarray, shape=shape)
    loss = categorical_crossentropy(scores, labels)
    assert len(loss) == 2
