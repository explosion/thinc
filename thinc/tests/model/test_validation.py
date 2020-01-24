import pytest
from thinc.api import chain, ReLu, reduce_max, Softmax, with_ragged
from thinc.api import ParametricAttention, list2ragged, reduce_sum
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


def test_validation_complex():
    good_model = chain(list2ragged(), reduce_sum(), ReLu(12, dropout=0.5), ReLu(1))
    X = [good_model.ops.xp.zeros((4, 75), dtype="f")]
    Y = good_model.ops.xp.zeros((1,), dtype="f")
    good_model.initialize(X, Y)
    good_model.predict(X)

    bad_model = chain(
        list2ragged(),
        reduce_sum(),
        ReLu(12, dropout=0.5),
        # ERROR: Why can't I attach a ReLu to an attention layer?
        ParametricAttention(12),
        ReLu(1),
    )
    with pytest.raises(DataValidationError):
        bad_model.initialize(X, Y)
