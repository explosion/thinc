from thinc.blas import Matrix
import numpy


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


def test_vec_max():
    array1 = numpy.asarray([[5, 10, 20], [40, 50, 60]])
    me = Matrix.from_array(array1)
    assert max(me) == max(array1.flatten())
    assert me.max() == max(array1.flatten())


def test_vec_sum():
    array1 = numpy.asarray([[5, 10, 20], [40, 50, 60]])
    me = Matrix.from_array(array1)
    assert sum(me) == sum(array1.flatten())
    assert me.sum() == sum(array1.flatten())


def test_matrix_scalar_iadd():
    array1 = numpy.asarray([[5, 10, 20], [40, 50, 60]])
    me = Matrix.from_array(array1)
    me += 5
    array1 += 5
    assert me == array1.flatten(), list(me)


def test_matrix_scalar_ipow():
    array1 = numpy.asarray([[5, 10, 20], [40, 50, 60]])
    me = Matrix.from_array(array1)
    me **= 2.0
    array1 **= 2.0
    assert me == array1.flatten(), list(me)


def test_matrix_scalar_imul():
    array1 = numpy.asarray([[5, 10, 20], [40, 50, 60]])
    me = Matrix.from_array(array1)
    me *= 5
    array1 *= 5
    assert me == array1.flatten(), list(me)


def test_mat_mat_iadd():
    array1 = numpy.asarray([[5, 10, 20], [40, 50, 60]])
    array2 = numpy.asarray([[1, 2, 3], [4, 5, 6]])
    me = Matrix.from_array(array1)
    you = Matrix.from_array(array2)
    array1 += array2
    me += you
    for i in range(me.nr_row * me.nr_col):
        print(i, me[i])
    assert me == array1.flatten()
