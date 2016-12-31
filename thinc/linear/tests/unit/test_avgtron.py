import pytest
import numpy
from numpy.testing import assert_allclose
from cytoolz import concat

from ...avgtron import AveragedPerceptron


@pytest.fixture
def templates():
    return (
        (10,),
        (2,),
        (10, 2),
        (689,)
    )


@pytest.fixture
def atoms(templates):
    atoms = numpy.zeros((max(concat(templates)),), dtype='uint64')
    atoms[10] = 100
    atoms[2] = 50009
    return atoms


@pytest.fixture
def nr_class():
    return 6


@pytest.fixture
def model(templates, nr_class):
    return AveragedPerceptron(templates, nr_out=nr_class)


def test_init(templates, model):
    assert model.nr_feat == len(templates) + 1
    

def test_call(model, atoms):
    scores = model(atoms)
    assert isinstance(scores, numpy.ndarray)
    assert scores.shape == (model.nr_out,)
    assert not numpy.isnan(scores.sum())


@pytest.mark.skip
def test_predict_batch(model, atoms):
    pass
    

def test_update_scores_match_call(model, atoms):
    scores_via_update, finish_update = model.begin_update(atoms)
    scores_via_call = model(atoms)
    assert_allclose(scores_via_update, scores_via_call)


def test_finish_update_executes(model, atoms):
    scores, finish_update = model.begin_update(atoms)
    gradient = numpy.zeros(scores.shape, dtype=scores.dtype)
    gradient[0] = 1
    gradient[1] = -2
    finish_update(gradient)
