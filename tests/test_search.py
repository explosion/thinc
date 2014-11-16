from __future__ import unicode_literals
from . import c_test_search as c


def test_init():
    c.test_init(5, 4)


def test_initialize():
    c.test_initialize(5, 4, 8)


def test_initialize_extra():
    c.test_initialize_extra(2, 1, 8, 'dingo')


def test_transition():
    c.test_transition()
