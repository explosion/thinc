# encoding: utf8
from __future__ import unicode_literals
import pytest
from flexmock import flexmock
from hypothesis import given, strategies
import abc

from .... import base


@pytest.mark.parametrize('new_name', ['mymodel', 'layer', 'basic', '', '漢字'])
def test_name_override(new_name):
    control = base.Model()
    assert control.name == 'model'
    model = base.Model(name=new_name)
    assert model.name == new_name
    assert model.name != 'model'
    control = base.Model()
    assert control.name == 'model'


@pytest.mark.parametrize('new_device', ['gpu', 'gpu1', 'foreign'])
def test_device_override(new_device):
    control = base.Model()
    assert control.device == 'cpu'
    model = base.Model(device=new_device)
    assert model.device == new_device
    assert model.device != 'cpu'
    control = base.Model()
    assert control.device == 'cpu'


def test_add_child_layer_instances():
    control = base.Model()
    assert len(control.layers) == 0
    model = base.Model(None, None,
                layers=(base.Model(name='child1'), base.Model(name='child2')))
    assert len(model.layers) == 2
    assert model.layers[0].name == 'child1'
    assert model.layers[1].name == 'child2'
    assert model.name == 'model'
    assert len(model.layers[0].layers) == 0
   
