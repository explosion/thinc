from thinc.api import NumpyOps, get_ops
from thinc.compat import has_apple_ops


def test_mps_ops_inherits_apple_ops():
    ops = get_ops("mps")
    assert isinstance(ops, NumpyOps)
    if has_apple_ops:
        # We can't import AppleOps directly, because its' not
        # available on non-Darwin systems.
        assert "AppleOps" in [base.__name__ for base in type(ops).__bases__]
