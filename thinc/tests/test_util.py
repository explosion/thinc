import pytest
from ..neural.util import get_ops
from ..neural.ops import NumpyOps, CupyOps


def test_get_ops():
    ops = get_ops('numpy')
    assert isinstance(ops, NumpyOps)
    ops = get_ops('cpu')
    assert isinstance(ops, NumpyOps)
    ops = get_ops('cupy')
    assert isinstance(ops, CupyOps)
    ops = get_ops('gpu')
    assert isinstance(ops, CupyOps)
    with pytest.raises(ValueError):
        ops = get_ops('blah')
