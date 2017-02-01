'''Test that about.py imports correctly and has expected attributes.'''

from ... import about


def test_about_attrs():
    about.__name__
    about.__version__
    about.__summary__
    about.__uri__
    about.__author__
    about.__email__
    about.__license__
    about.__title__
    about.__release__
