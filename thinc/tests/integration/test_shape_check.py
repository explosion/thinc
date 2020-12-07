# coding: utf8
from __future__ import unicode_literals

import pytest
import numpy

from thinc.neural._classes.affine import Affine
from thinc.exceptions import ShapeMismatchError


def test_mismatched_shapes_raise_ShapeMismatchError():
    X = numpy.ones((3, 4))
    model = Affine(10, 5)
    with pytest.raises(ShapeMismatchError):
        model.predict(X)
