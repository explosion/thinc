# coding: utf-8
from __future__ import unicode_literals

import pytest
from mock import MagicMock
from numpy import ndarray

from ... import check
from ...neural._classes.model import Model
from ...exceptions import UndefinedOperatorError, DifferentLengthError
from ...exceptions import ExpectedTypeError, ShapeMismatchError
from ...exceptions import ConstraintError, OutsideRangeError


@pytest.fixture
def model():
    return Model()


@pytest.fixture
def dummy(*args, **kwargs):
    def _dummy(*args, **kwargs):
        return None
    return dummy


@pytest.mark.parametrize('operator', ['+'])
def test_check_operator_is_defined_passes(model, dummy, operator):
    checker = check.operator_is_defined(operator)
    checked = checker(dummy)
    with Model.define_operators({'+': None}):
        checked(model, None)


def test_check_operator_is_defined_type_fails(model, dummy):
    checker = check.operator_is_defined('')
    checked = checker(dummy)
    with pytest.raises(ExpectedTypeError):
        checked(None)


@pytest.mark.parametrize('operator', ['+'])
def test_check_operator_is_defined_fails(model, dummy, operator):
    checker = check.operator_is_defined(operator)
    checked = checker(dummy)
    with pytest.raises(UndefinedOperatorError):
        checked(model, None)


@pytest.mark.parametrize('args', [[(1, 2, 3), (4, 5, 6)]])
def test_check_equal_length_passes(args):
    check.equal_length(args)


@pytest.mark.parametrize('args', [[(1, 2, 3), (4, 5, 6, 7), (0)],
                                  ['hello', 'worlds'],
                                  [(True, False), ()]])
def test_check_equal_length_fails(args):
    with pytest.raises(DifferentLengthError):
        check.equal_length(*args)


@pytest.mark.parametrize('args', [[(1, 2, 3), True],
                                  ['hello', 'world', 14],
                                  [(True, False), None]])
def test_check_equal_length_type_fails(args):
    with pytest.raises(ExpectedTypeError):
        check.equal_length(*args)


def test_check_has_shape_passes():
    mock = MagicMock(spec=ndarray)
    check.has_shape(0, [mock], None)


@pytest.mark.parametrize('arg', [True, None, 14])
def test_check_has_shape_fails(arg):
    with pytest.raises(ExpectedTypeError):
        check.has_shape(None, 0, [arg], None)


@pytest.mark.parametrize('shape', [[1, 2], [3], [4, 'hello']])
@pytest.mark.parametrize('arg', [['world', 2, 3]])
def test_check_has_shape_mismatch_fails(arg, shape):
    mock_shape = MagicMock(spec=ndarray)
    mock_shape.__len__.return_value = len(shape)
    mock_shape.__iter__.return_value = shape
    mock_shape.return_value = shape
    mock_arg = MagicMock(spec=ndarray, shape=[10, 20])
    mock_arg.__iter__.return_value = arg
    with pytest.raises(ShapeMismatchError):
        check.has_shape(mock_shape, 0, [mock_arg], None)


@pytest.mark.parametrize('arg', [(1, 2, 3)])
def test_check_is_shape_passes(arg):
    check.is_shape(0, [arg], None)


@pytest.mark.parametrize('arg', [None, 14, (-1, 0, 1), (14, 'hello')])
def test_check_is_shape_fails(arg):
    with pytest.raises(ExpectedTypeError):
        check.is_shape(0, [arg], None)


@pytest.mark.parametrize('arg', [(1, 2, 3)])
def test_check_is_sequence_passes(arg):
    check.is_sequence(0, [arg], None)


@pytest.mark.parametrize('arg', [True, None, 14])
def test_check_is_sequence_fails(arg):
    with pytest.raises(ExpectedTypeError):
        check.is_sequence(0, [arg], None)


@pytest.mark.parametrize('arg', [1.0])
def test_check_is_float_passes(arg):
    check.is_float(0, [arg], None)


@pytest.mark.parametrize('arg', [True, None, (1, 2, 3), {'foo': 'bar'}])
def test_check_is_float_fails(arg):
    with pytest.raises(ExpectedTypeError):
        check.is_float(0, [arg], None)


@pytest.mark.parametrize('low,high', [(1.0, 12.0)])
def test_check_is_float_min_max_passes(low, high):
    check.is_float(0, [low], None, min=low)
    check.is_float(0, [high], None, max=high)


@pytest.mark.parametrize('low,high', [(1.0, 12.0), (123.456, 789.0)])
def test_check_is_float_min_fails(low, high):
    with pytest.raises(OutsideRangeError):
        check.is_float(0, [low], None, min=high)


@pytest.mark.parametrize('low,high', [(1.0, 12.0), (123.456, 789.0)])
def test_check_is_float_max_fails(low, high):
    with pytest.raises(OutsideRangeError):
        check.is_float(0, [high], None, max=low)


@pytest.mark.parametrize('arg', [1])
def test_check_is_int_passes(arg):
    check.is_int(0, [arg], None)


@pytest.mark.parametrize('arg', ['hello', None, (1, 2, 3), {'foo': 'bar'}])
def test_check_is_int_fails(arg):
    with pytest.raises(ExpectedTypeError):
        check.is_int(0, [arg], None)


@pytest.mark.parametrize('low,high', [(1, 12)])
def test_check_is_int_min_max_passes(low, high):
    check.is_int(0, [low], None, min=low)
    check.is_int(0, [high], None, max=high)


@pytest.mark.parametrize('low,high', [(1, 12), (123, 789)])
def test_check_is_int_min_fails(low, high):
    with pytest.raises(OutsideRangeError):
        check.is_int(0, [low], None, min=high)


@pytest.mark.parametrize('low,high', [(1, 12), (123, 789)])
def test_check_is_int_max_fails(low, high):
    with pytest.raises(OutsideRangeError):
        check.is_int(0, [high], None, max=low)


def test_check_is_array_passes():
    mock = MagicMock(spec=ndarray)
    check.is_array(0, [mock], None)


@pytest.mark.parametrize('arg', ['hello', (1, 2, 3), True, None])
def test_check_is_array_fails(arg):
    with pytest.raises(ExpectedTypeError):
        check.is_array(0, [arg], None)


def test_check_is_int_array_passes():
    mock = MagicMock(spec=ndarray, dtype='i')
    check.is_int_array(0, [mock], None)


@pytest.mark.parametrize('arg', ['hello', (1, 2, 3), True, None])
def test_check_is_int_array_fails(arg):
    with pytest.raises(ExpectedTypeError):
        check.is_int_array(0, [arg], None)


def test_check_is_int_array_ndarray_fails():
    mock = MagicMock(dtype='x')
    with pytest.raises(ExpectedTypeError):
        check.is_int_array(0, [mock], None)


def test_check_arg_passes():
    checker = check.arg(0, check.is_int)
    checked = checker(dummy)
    checked(0, [1], True)


def test_check_args_passes():
    checker = check.args(check.is_int)
    checked = checker(dummy)
    checked(0, [1], None)
