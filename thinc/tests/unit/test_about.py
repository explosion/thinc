# coding: utf8
from __future__ import unicode_literals

from thinc import about


def test_about_attrs():
    """Test that about.py imports correctly and has expected attributes."""
    about.__name__
    about.__version__
    about.__summary__
    about.__uri__
    about.__author__
    about.__email__
    about.__license__
    about.__title__
    about.__release__
