# Necessary for some side-effects in Cython. Not sure I understand.
import numpy  # noqa: F401

from .about import __version__  # noqa: F401
from ._registry import registry  # noqa: F401
