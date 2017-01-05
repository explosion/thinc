# encoding: utf8
from __future__ import unicode_literals
import pytest
from flexmock import flexmock
from hypothesis import given, strategies
import abc

from .... import base
from .... import ops


@pytest.fixture
def model_with_no_args():
    flexmock(base.Model)
    model = base.Model()
    return model


def test_Model_defaults_to_name_model(model_with_no_args):
    assert model_with_no_args.name == 'model'


def test_Model_defaults_to_cpu(model_with_no_args):
    assert isinstance(model_with_no_args.ops, ops.NumpyOps)


def test_Model_defaults_to_no_layers(model_with_no_args):
    assert model_with_no_args.layers == []


def test_Model_defaults_to_no_param_descripions(model_with_no_args):
    assert list(model_with_no_args.describe_params) == []


def test_Model_defaults_to_no_output_shape(model_with_no_args):
    assert model_with_no_args.output_shape == None
 

def test_Model_defaults_to_no_input_shape(model_with_no_args):
    assert model_with_no_args.input_shape == None

#
#def test_Model_defaults_to_0_size(model_with_no_args):
#    assert model_with_no_args.size == 0
#
#
#def test_Model_defaults_to_no_params(model_with_no_args):
#    assert model_with_no_args.params is None

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
    flexmock(base.util)
    base.util.should_receive('get_ops').and_return(ops.NumpyOps())
    model = base.Model(device=new_device)
    assert model.device == new_device
    assert model.device != 'cpu'
    control = base.Model()
    assert control.device == 'cpu'


def test_add_child_layer_instances():
    control = base.Model()
    assert len(control.layers) == 0
    model = base.Model(
                base.Model(name='child1'),
                base.Model(name='child2'))
    assert len(model.layers) == 2
    assert model.layers[0].name == 'child1'
    assert model.layers[1].name == 'child2'
    assert model.name == 'model'
    assert len(model.layers[0].layers) == 0
   

#
#@given(strategies.integers(min_value=0))
#def test_model_with_unk_input_shape_scalar_output_shape(scalar_output_shape):
#    '''This jointly tests init, _update_defaults, and _args2kwargs.'''
#    flexmock(base.Model)
#    flexmock(base.util)
#    base.util.should_receive('get_ops').with_args('cpu').and_return('cpu-ops')
#    model = base.Model(scalar_output_shape)
#
#    assert model.output_shape == scalar_output_shape
#    assert model.input_shape is None
