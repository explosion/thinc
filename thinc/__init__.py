# Necessary for some side-effects in Cython. Not sure I understand.
import numpy

from .about import __version__
from .util import prefer_gpu, require_gpu
from ._registry import registry
