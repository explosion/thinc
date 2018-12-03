# coding: utf8
from __future__ import unicode_literals

import pytest
from srsly import cloudpickle as pickle

from ...api import with_flatten
from ...v2v import Affine


@pytest.fixture
def affine():
    return Affine(5, 3)


def test_pickle_with_flatten(affine):
    Xs = [affine.ops.allocate((2, 3)), affine.ops.allocate((4, 3))]
    model = with_flatten(affine)
    pickled = pickle.dumps(model)
    loaded = pickle.loads(pickled)
    Ys = loaded(Xs)
    assert len(Ys) == 2
    assert Ys[0].shape == (Xs[0].shape[0], affine.nO)
    assert Ys[1].shape == (Xs[1].shape[0], affine.nO)
