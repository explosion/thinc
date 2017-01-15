import pytest

from .. import exceptions as e


def test_shape_error():
    raise_if(e.ShapeError.dimensions_mismatch(10, 20, 'inside test'))


def raise_if(e):
    if e is not None:
        raise e.with_traceback(self.tb)

