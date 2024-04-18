from typing import TYPE_CHECKING

import numpy

from .. import registry
from ..compat import has_apple_ops
from .numpy_ops import NumpyOps
from .ops import Ops

if TYPE_CHECKING:
    # Type checking does not work with dynamic base classes, since MyPy cannot
    # determine against which base class to check. So, always derive from Ops
    # during type checking.
    _Ops = Ops
else:
    if has_apple_ops:
        from .apple_ops import AppleOps

        _Ops = AppleOps
    else:
        _Ops = NumpyOps


@registry.ops("MPSOps")
class MPSOps(_Ops):
    """Ops class for Metal Performance shaders."""

    name = "mps"
    xp = numpy
