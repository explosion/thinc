import pytest

from ...._classes.model import Model


def test_bind_plus():
    with Model.define_operators({'+': lambda a, b: (a.name, b.name)}):
        m = Model(name='a') + Model(name='b')
        assert m == ('a', 'b')

def test_plus_chain():
    with Model.define_operators({'+': lambda a, b: a}):
        m = Model(name='a') + Model(name='b') + Model(name='c') + Model(name='d')
        assert m.name == 'a'
