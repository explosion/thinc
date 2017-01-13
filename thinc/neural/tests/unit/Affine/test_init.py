# encoding: utf8
from __future__ import unicode_literals
import pytest
from mock import Mock, patch
from hypothesis import given, strategies
import abc

from ...._classes.affine import Affine
from ....ops import NumpyOps


@pytest.fixture
def model():
    orig_desc = dict(Affine.descriptions)
    orig_on_init = list(Affine.on_init_hooks)
    Affine.descriptions = {
        name: Mock(desc) for (name, desc) in Affine.descriptions.items()
    }
    Affine.on_init_hooks = [Mock(hook) for hook in Affine.on_init_hooks]
    model = Affine()
    Affine.descriptions = dict(orig_desc)
    Affine.on_init_hooks = orig_on_init
    return model


def test_Affine_default_name(model):
    assert model.name == 'affine'


def test_Affine_calls_default_descriptions(model):
    assert len(model.descriptions) == 7
    for name, desc in model.descriptions.items():
        desc.assert_called()
    assert 'nB' in model.descriptions
    assert 'nI' in model.descriptions
    assert 'nO' in model.descriptions
    assert 'W' in model.descriptions
    assert 'b' in model.descriptions
    assert 'd_W' in model.descriptions
    assert 'd_b' in model.descriptions


def test_Affine_calls_init_hooks(model):
    for hook in model.on_init_hooks:
        hook.assert_called()


def test_Affine_dimensions_on_data():
    X = Mock()
    X.shape = Mock()
    X.shape.__getitem__ = Mock()
    y = Mock()
    y.max = Mock()
    model = Affine()
    with model.begin_training(X, y):
        pass
    X.shape.__getitem__.assert_called_with(0)
    y.max.assert_called_with()
