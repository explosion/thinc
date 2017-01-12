# encoding: utf8
from __future__ import unicode_literals
import pytest
from flexmock import flexmock
from hypothesis import given, strategies
import abc

from ...._classes import model as base
from ....ops import NumpyOps


@pytest.fixture
def model_with_no_args():
    model = base.Model()
    return model


def test_Model_defaults_to_name_model(model_with_no_args):
    assert model_with_no_args.name == 'model'


def test_changing_instance_name_doesnt_change_class_name():
    model = base.Model()
    assert model.name != 'changed'
    model.name = 'changed'
    model2 = base.Model()
    assert model2.name != 'changed'


def test_changing_class_name_doesnt_change_default_instance_name():
    model = base.Model()
    assert model.name != 'changed'
    base.Model.name = 'changed'
    assert model.name != 'changed'
    # Reset state
    base.Model.name = 'model'

def test_changing_class_name_doesnt_changes_nondefault_instance_name():
    model = base.Model(name='nondefault')
    assert model.name == 'nondefault'
    base.Model.name = 'changed'
    assert model.name == 'nondefault'


def test_Model_defaults_to_cpu(model_with_no_args):
    assert isinstance(model_with_no_args.ops, NumpyOps)


def test_models_get_different_ids(model_with_no_args):
    model1 = base.Model()
    model2 = base.Model()
    assert model1.id != model2.id
