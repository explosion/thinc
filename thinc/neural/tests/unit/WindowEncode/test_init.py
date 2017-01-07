import pytest

from ...._classes.window_encode import MaxoutWindowEncode
from ....ops import NumpyOps


def test_init_succeeds():
    model = MaxoutWindowEncode(10)


def test_init_defaults_and_overrides():
    model = MaxoutWindowEncode(10)
    assert model.nr_piece == MaxoutWindowEncode.nr_piece
    assert model.nr_feat == MaxoutWindowEncode.nr_feat
    assert model.nr_out == 10
    assert model.nr_in == None
    model = MaxoutWindowEncode(10, nr_piece=MaxoutWindowEncode.nr_piece-1)
    assert model.nr_piece == MaxoutWindowEncode.nr_piece-1
    model = MaxoutWindowEncode(10, nr_feat=MaxoutWindowEncode.nr_feat-1)
    assert model.nr_feat == MaxoutWindowEncode.nr_feat-1
