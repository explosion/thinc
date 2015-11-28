from thinc.blas import Matrix


def test_init():
    matrix = Matrix(5, 5)


def test_set_get():
    matrix = Matrix(5, 5)
    assert matrix.get(0, 1) == 0.0
    matrix.set(0, 1, 0.5)
    assert matrix.get(0, 1) == 0.5
    matrix.set(0, 1, 0.0)
    assert matrix.get(0, 1) == 0.0
    matrix.set(0, 0, 0.5)
    assert matrix.get(0, 1) == 0.0
    assert matrix.get(0, 0) == 0.5


def test_dot_bias():
    me = Matrix(1, 3)
    W = Matrix(3, 3)
    b = Matrix(1, 3)

    me.set(0, 0, 1.0)
    me.set(0, 1, 1.0)
    me.set(0, 2, 1.0)

    W.set(0, 0, 0)
    W.set(0, 1, 0)
    W.set(0, 2, 0)

    W.set(1, 0, 1)
    W.set(1, 1, 1)
    W.set(1, 2, 1)

    W.set(2, 0, 2)
    W.set(2, 1, 2)
    W.set(2, 2, 2)

    b.set(0, 0, 0.25)
    b.set(0, 1, 0.5)
    b.set(0, 2, 0.75)

    output = me.dot_bias(W, b)

    assert output.get(0, 0) == 0.25
    assert output.get(0, 1) == 3.5
    assert output.get(0, 2) == 6.75


def test_iadd():
    me = Matrix(1, 3)
    you = Matrix(1, 3)

    me.set(0, 0, 1.0)
    me.set(0, 1, 2.0)
    me.set(0, 2, 3.0)

    you.set(0, 0, 0.25)
    you.set(0, 1, 0.5)
    you.set(0, 2, 0.75)

    me += you
    assert me.get(0, 0) == 1.25
    assert me.get(0, 1) == 2.5
    assert me.get(0, 2) == 3.75
