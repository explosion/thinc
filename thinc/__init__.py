# Necessary for some side-effects in Cython. Not sure I understand.
import numpy

from .about import __version__
from .config import registry
from .util import require_cpu


# default to AppleOps instead of NumpyOps
require_cpu()
