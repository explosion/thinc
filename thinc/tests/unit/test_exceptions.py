import pytest

from ... import check
from ...neural._classes.model import Model
from ...exceptions import UndefinedOperatorError, DifferentLengthError, ExpectedTypeError


@pytest.fixture
def model():
    return Model()


@pytest.fixture
def dummy():
    def _dummy(*args, **kwargs):
        return None
    return dummy


@pytest.mark.parametrize('operator', ['+'])
def test_check_operator_is_defined(model, dummy, operator):
    checker = check.operator_is_defined(operator)
    checked = checker(dummy)
    with pytest.raises(UndefinedOperatorError):
        checked(model, None)


@pytest.mark.parametrize('args', [[(1, 2, 3), (4, 5, 6, 7), (0)],
                                  ['hello', 'worlds'],
                                  [(True, False), ()]])
def test_check_equal_length_different_length(args):
    with pytest.raises(DifferentLengthError):
        check.equal_length(*args)


@pytest.mark.parametrize('args', [[(1, 2, 3), True],
                                  ['hello', 'world', 14],
                                  [(True, False), None]])
def test_check_equal_length_type_error(args):
    with pytest.raises(ExpectedTypeError):
        check.equal_length(*args)


@pytest.mark.parametrize('arg', [True, None, 14])
def test_check_is_sequence(arg):
    args = [(1, 2, 3), arg]
    with pytest.raises(ExpectedTypeError):
        check.is_sequence(1, args, None)


@pytest.mark.parametrize('arg', [True, None, (1, 2, 3), {'foo': 'bar'}])
def test_check_is_float(arg):
    args = [1.0, arg]
    with pytest.raises(ExpectedTypeError):
        check.is_float(1, args, None)


@pytest.mark.parametrize('low,high', [(1.0, 12.0), (123.456, 789.0)])
def test_check_is_float_min_max(low, high):
    with pytest.raises(ValueError):
        check.is_float(1, [1.0, low], None, min=high)

    with pytest.raises(ValueError):
        check.is_float(1, [1.0, high], None, max=low)


@pytest.mark.parametrize('arg', ['hello', None, (1, 2, 3), {'foo': 'bar'}])
def test_check_is_int(arg):
    args = [1, arg]
    with pytest.raises(ExpectedTypeError):
        check.is_int(1, args, None)


@pytest.mark.parametrize('low,high', [(1, 12), (123, 789)])
def test_check_is_int_min_max(low, high):
    with pytest.raises(ValueError):
        check.is_int(1, [1, low], None, min=high)

    with pytest.raises(ValueError):
        check.is_int(1, [1, high], None, max=low)
