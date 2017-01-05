# encoding: utf8
from __future__ import unicode_literals
import pytest
from flexmock import flexmock
from hypothesis import given, strategies
import abc

from .... import vec2vec
from ....ops import NumpyOps


@pytest.fixture
def model_with_no_args():
    model = vec2vec.Affine(ops=NumpyOps())
    return model


def test_Affine_default_name(model_with_no_args):
    assert model_with_no_args.name == 'affine'


def test_Affine_defaults_to_cpu(model_with_no_args):
    assert isinstance(model_with_no_args.ops, NumpyOps)


def test_Affine_defaults_to_no_layers(model_with_no_args):
    assert model_with_no_args.layers == []


def test_Affine_defaults_to_param_descriptions(model_with_no_args):
    W_desc, b_desc = model_with_no_args.describe_params
    xavier_init = model_with_no_args.ops.xavier_uniform_init
    assert W_desc == ('W-affine', (None, None), xavier_init)
    assert b_desc == ('b-affine', (None,), None)


def test_Model_defaults_to_no_output_shape(model_with_no_args):
    assert model_with_no_args.output_shape == None
 

def test_Model_defaults_to_no_input_shape(model_with_no_args):
    assert model_with_no_args.input_shape == None


def test_Model_defaults_to_0_size(model_with_no_args):
    assert model_with_no_args.size == None
