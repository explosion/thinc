# coding: utf8
from __future__ import unicode_literals

from thinc.neural._classes.maxout import Maxout
from thinc.api import chain


def test_simple_model_roundtrip_bytes():
    model = Maxout(5, 10, pieces=2)
    model.b += 1
    data = model.to_bytes()
    model.b -= 1
    model = model.from_bytes(data)
    assert model.b[0, 0] == 1


def test_multi_model_roundtrip_bytes():
    model = chain(Maxout(5, 10, pieces=2), Maxout(2, 3))
    model._layers[0].b += 1
    model._layers[1].b += 2
    data = model.to_bytes()
    model._layers[0].b -= 1
    model._layers[1].b -= 2
    model = model.from_bytes(data)
    assert model._layers[0].b[0, 0] == 1
    assert model._layers[1].b[0, 0] == 2


def test_multi_model_load_missing_dims():
    model = chain(Maxout(5, 10, pieces=2), Maxout(2, 3))
    model._layers[0].b += 1
    model._layers[1].b += 2
    data = model.to_bytes()

    model2 = chain(Maxout(5), Maxout())
    model2 = model2.from_bytes(data)
    assert model2._layers[0].b[0, 0] == 1
    assert model2._layers[1].b[0, 0] == 2
