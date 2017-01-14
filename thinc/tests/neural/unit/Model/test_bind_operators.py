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


@pytest.mark.parametrize('op', '+ - * @ / // % ** << >> & ^ |'.split())
def test_all_operators(op):
    m1 = Model(name='a')
    m2 = Model(name='b')
    with Model.define_operators({op: lambda a, b: a.name + b.name}):
        if op == '+':
            value = m1 + m2
        else:
            with pytest.raises(TypeError):
                value = m1 + m2
        if op == '-':
            value = m1 - m2
        else:
            with pytest.raises(TypeError):
                value = m1 - m2

        if op == '*':
            value = m1 * m2
        else:
            with pytest.raises(TypeError):
                value = m1 * m2

        if op == '@':
            value = m1.__matmul__(m2) # Be kind to Python 2...
        else:
            with pytest.raises(TypeError):
                value = m1.__matmul__(m2)

        if op == '/':
            value = m1 / m2
        else:
            with pytest.raises(TypeError):
                value = m1 / m2

        if op == '//':
            value = m1 // m2
        else:
            with pytest.raises(TypeError):
                value = m1 // m2
        if op == '^':
            value = m1 ^ m2
        else:
            with pytest.raises(TypeError):
                value = m1 ^ m2
        if op == '%':
            value = m1 % m2
        else:
            with pytest.raises(TypeError):
                value = m1 % m2
        if op == '**':
            value = m1 ** m2
        else:
            with pytest.raises(TypeError):
                value = m1 ** m2
        if op == '<<':
            value = m1 << m2
        else:
            with pytest.raises(TypeError):
                value = m1 << m2
        if op == '>>':
            value = m1 >> m2
        else:
            with pytest.raises(TypeError):
                value = m1 >> m2
        if op == '&':
            value = m1 & m2
        else:
            with pytest.raises(TypeError):
                value = m1 & m2
        if op == '^':
            value = m1 ^ m2
        else:
            with pytest.raises(TypeError):
                value = m1 ^ m2
        if op == '|':
            value = m1 | m2
        else:
            with pytest.raises(TypeError):
                value = m1 | m2
    assert Model._operators == {}
