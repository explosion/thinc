# coding: utf8
from __future__ import unicode_literals

import pytest
import numpy
import numpy.linalg

# from ...neural._classes.difference import word_movers_similarity


@pytest.fixture
def N1():
    return 5


@pytest.fixture
def N2():
    return 3


@pytest.fixture
def ndim():
    return 2


@pytest.fixture
def mat1(N1, ndim):
    mat = numpy.ones((N1, ndim))
    for i in range(N1):
        mat[i] /= numpy.linalg.norm(mat[i])
    return mat


@pytest.fixture
def mat2(N2, ndim):
    mat = numpy.ones((N2, ndim))
    for i in range(N2):
        mat[i] /= numpy.linalg.norm(mat[i])
    return mat


def cosine_similarity(vec1_vec2):
    # Assume vectors are normalized
    vec1, vec2 = vec1_vec2

    def backward(d_sim, sgd=None):
        if d_sim.ndim == 1:
            d_sim = d_sim.reshape((d_sim.shape[0], 1))
        print(vec1.shape, d_sim.shape)
        print(vec1.shape, d_sim.shape)
        return (vec2 * d_sim, vec1 * d_sim)

    dotted = (vec1 * vec2).sum(axis=1)
    return dotted, backward


# def test_word_movers_similarity_unit_matrices(mat1, mat2):
#    sim, backward = word_movers_similarity(mat1, mat2)
#    assert_allclose(sim, 1.0)
#    d_mat1, d_mat2 = backward(0.0, None)
#    assert d_mat1.shape == mat1.shape
#    assert d_mat2.shape == mat2.shape
#
#
# def test_gradient(mat1, mat2):
#    mat1[0] = 10.0
#    mat2[-1] = 10.0
#    mat1[1] = -2.
#    mat2[0] = -2
#    sim, backward = word_movers_similarity(mat1, mat2)
#    d_mat1, d_mat2 = backward(-1.0)
#    assert d_mat1[0, -1] != 0.
#    assert d_mat1[0, 0] == (-1./(mat1.shape[0]+mat2.shape[0])) * 10.
