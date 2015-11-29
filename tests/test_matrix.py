from thinc.blas import Matrix
import numpy
import numpy.testing
import pytest


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


def test_mat_vec_dot():
    np_mat = numpy.asarray([[5, 10, 20], [40, 50, 60]])
    np_vec = numpy.asarray([2, 4, 8])
    mat = Matrix.from_array(np_mat)
    vec = Matrix.from_array(np_vec)
    output = mat.dot(vec)
    assert output == np_mat.dot(np_vec)


def test_vec_vec_add():
    np_vec1 = numpy.asarray([5, 10, 20])
    np_vec2 = numpy.asarray([2, 4, 8])
    vec1 = Matrix.from_array(np_vec1)
    vec2 = Matrix.from_array(np_vec2)
    vec1 += vec2
    np_vec1 += np_vec2
    assert list(vec1) == list(np_vec1.flatten())


def test_vec_add():
    np_vec1 = numpy.asarray([5, 10, 20])
    vec1 = Matrix.from_array(np_vec1)
    np_vec1 += 10
    vec1 += 10
    assert list(vec1) == list(np_vec1.flatten())


def test_vec_exp():
    np_vec1 = numpy.asarray([5, 10, 20])
    vec1 = Matrix.from_array(np_vec1)
    np_vec1 = numpy.exp(np_vec1)
    vec1.exp()
    numpy.testing.assert_allclose(list(vec1), list(np_vec1.flatten()))


def test_vec_div():
    np_vec1 = numpy.asarray([5.0, 10.0, 20.0])
    vec1 = Matrix.from_array(np_vec1)
    np_vec1 /= 2.0
    vec1 /= 2.0
    numpy.testing.assert_allclose(list(vec1), list(np_vec1.flatten()))


def test_mat_mat_add_outer():
    np_x = numpy.asarray([2, 4, 8])
    np_y = numpy.asarray([10, 20])
    np_mat = numpy.zeros(shape=(3, 2))
    mat = Matrix.from_array(np_mat)
    x = Matrix.from_array(np_x)
    y = Matrix.from_array(np_y)

    mat.add_outer(x, y)
    np_mat += numpy.outer(np_x, np_y)
    assert list(mat) == list(np_mat.flatten())
