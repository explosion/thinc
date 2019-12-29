import pytest
from thinc.backends import NumpyOps, CupyOps
from thinc.util import get_ops


def test_get_ops():
    Ops = get_ops("numpy")
    Ops is NumpyOps
    Ops = get_ops("cpu")
    assert Ops is NumpyOps
    Ops = get_ops("cupy")
    assert Ops is CupyOps
    Ops = get_ops("gpu")
    assert Ops is CupyOps
    with pytest.raises(ValueError):
        Ops = get_ops("blah")
