import pytest
import numpy

from ...neural._classes.affine import Affine
from ...exceptions import UndefinedOperatorError, DifferentLengthError
from ...exceptions import ExpectedTypeError, ShapeMismatchError


@pytest.mark.xfail
def test_mismatched_shapes_raise_ShapeError():
    X = numpy.ones((3, 4), dtype='f')
    model = Affine(10, 5)
    with pytest.raises(ShapeMismatchError):
        y = model.predict(X)

