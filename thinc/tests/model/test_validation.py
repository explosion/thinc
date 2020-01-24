import pytest
from thinc.api import chain, ReLu, reduce_max, Softmax, with_ragged
from thinc.util import DataValidationError
import numpy


def test_validation():
    model = chain(ReLu(10), ReLu(10), with_ragged(reduce_max()), Softmax())
    with pytest.raises(DataValidationError):
        model.initialize(X=model.ops.alloc_f2d(1, 10), Y=model.ops.alloc_f2d(1, 10))
    with pytest.raises(DataValidationError):
        model.initialize(X=model.ops.alloc_f3d(1, 10, 1), Y=model.ops.alloc_f2d(1, 10))
    with pytest.raises(DataValidationError):
        model.initialize(X=[model.ops.alloc_f2d(1, 10)], Y=model.ops.alloc_f2d(1, 10))
    # X = numpy.asarray([[0.1, 0.1], [-0.1, -0.1], [1.0, 1.0]], dtype="f")
    # model = with_padded(LSTM(1, 2)).initialize(X=X)
