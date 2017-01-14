import pytest
import numpy
from numpy.testing import assert_allclose
from cytoolz import concat


from ....neural._classes.window_encode import MaxoutWindowEncode
from ....neural.ops import NumpyOps


@pytest.fixture
def nr_out():
    return 5


@pytest.fixture
def ndim():
    return 3


@pytest.fixture
def ops():
    return NumpyOps()


@pytest.fixture
def model(ops, nr_out, ndim):
    model = MaxoutWindowEncode(nr_out=nr_out, nr_in=ndim, ops=ops)
    model.initialize_params()
    return model


@pytest.fixture
def ids():
    return [[0, 1, 0, 2]]


@pytest.fixture
def lengths(ids):
    return [len(seq) for seq in ids]


@pytest.fixture
def vectors(ndim, lengths):
    table = []
    for i in range(sum(lengths)):
        table.append([i+float(j) / ndim for j in range(ndim)])
    return numpy.asarray(table, dtype='f') 


@pytest.fixture
def gradients_BO(model, lengths):
    gradient = model.ops.allocate((sum(lengths), model.nr_out)) 
    for i in range(gradient.shape[0]):
        gradient[i] -= i
    return gradient

@pytest.fixture
def B(lengths):
    return sum(lengths)

@pytest.fixture
def I(ndim):
    return ndim

@pytest.fixture
def O(nr_out):
    return nr_out


@pytest.mark.xfail
def test_update_shape(B, I, O, model, ids, vectors, lengths, gradients_BO):
    assert gradients_BO.shape == (B, O)
    ids = list(concat(ids))
    fwd, finish_update = model.begin_update((ids, vectors, lengths))
    gradients_BI = finish_update(gradients_BO, optimizer=None)
    assert gradients_BI.shape == (B, I)



@pytest.mark.xfail
def test_zero_gradient_makes_zero_finetune(model):
    ids = [0]
    lengths = [1]
    vectors = numpy.asarray([[0., 0., 0.]], dtype='f')
    gradients_BO = numpy.asarray([[0., 0., 0., 0., 0.]], dtype='f')
    fwd, finish_update = model.begin_update((ids, vectors, lengths))
    gradients_BI = finish_update(gradients_BO, optimizer=None)
    assert_allclose(gradients_BI, [[0., 0., 0.]])



@pytest.mark.xfail
def test_negative_gradient_positive_weights_makes_negative_finetune(model):
    ids = [0]
    lengths = [1]
    W = model.W
    W.fill(1)
    vectors = numpy.asarray([[0., 0., 0.]], dtype='f')
    gradients_BO = numpy.asarray([[-1., -1., -1., -1., -1.]], dtype='f')
    fwd, finish_update = model.begin_update((ids, vectors, lengths))
    gradients_BI = finish_update(gradients_BO, optimizer=None)

    for val in gradients_BI.flatten():
        assert val < 0


@pytest.mark.xfail
def test_vectors_change_fine_tune(model):
    ids = [0]
    lengths = [1]
    W = model.W
    W.fill(1)
    gradients_BO = numpy.asarray([[-1., -1., -1., -1., -1.]], dtype='f')
    vec1 = numpy.asarray([[1., 1., 1.]], dtype='f')
    fwd, finish_update = model.begin_update((ids, vec1, lengths))
    grad1 = finish_update(gradients_BO, optimizer=None)
    
    vec2 = numpy.asarray([[2., 2., 2.]], dtype='f')
    fwd, finish_update = model.begin_update((ids, vec2, lengths))
    grad2 = finish_update(gradients_BO, optimizer=None)
    
    for val1, val2 in zip(vec1.flatten(), vec2.flatten()):
        assert val1 != val2
