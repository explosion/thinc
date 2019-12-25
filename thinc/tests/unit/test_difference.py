import pytest
import numpy
import numpy.linalg


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
