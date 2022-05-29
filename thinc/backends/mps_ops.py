import numpy

from .. import registry
from . import NumpyOps

try:
    from thinc_apple_ops import AppleOps

    _Ops = AppleOps
except:
    _Ops = NumpyOps


@registry.ops("MPSOps")
class MPSOps(_Ops):
    """Ops class for Metal Performance shaders."""

    name = "mps"
    xp = numpy
