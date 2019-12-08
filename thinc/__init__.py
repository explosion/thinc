# coding: utf8
from __future__ import unicode_literals

# Necessary for some side-effects in Cython. Not sure I understand.
import numpy  # noqa: F401

from .about import __name__, __version__  # noqa: F401
from ._registry import registry
