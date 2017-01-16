import pytest

from ... import check
from ...neural._classes.model import Model
from ...exceptions import UndefinedOperatorError


#@pytest.fixture
#def add_model():
#    return Model().__add__

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
