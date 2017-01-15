import pytest
import numpy

from ...neural._classes.model import Model
from ...exceptions import UndefinedOperatorError, DifferentLengthError
from ...exceptions import ExpectedTypeError, ShapeMismatchError


def test_mismatched_shapes_raise_ShapeError():
    X = numpy.ones((3, 4))
    model = Model(10, 5)
    with pytest.raises(ShapeMismatchError):
        y = model.begin_training(X, X)

