import pytest
import numpy
from numpy.testing import assert_allclose
from cytoolz import concat

from ....linear.avgtron import AveragedPerceptron


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
    

@pytest.mark.xfail
def test_call(model, atoms):
    scores = model(atoms)
    assert isinstance(scores, numpy.ndarray)
    assert scores.shape == (model.nr_out,)
    assert not numpy.isnan(scores.sum())


@pytest.mark.skip
def test_predict_batch(model, atoms):
    pass
    
@pytest.mark.xfail
def test_update_scores_match_call(model, atoms):
    atoms = numpy.expand_dims(atoms, 0)
    scores_via_update, finish_update = model.begin_update(atoms)
    scores_via_call = model(atoms[0])
    assert_allclose(scores_via_update[0], scores_via_call)


@pytest.mark.xfail
def test_finish_update_executes(model, atoms):
    atoms = numpy.expand_dims(atoms, 0)
    scores, finish_update = model.begin_update(atoms)
    assert scores.shape == (1, model.nr_out)
    labels = numpy.zeros(scores.shape[:1], dtype='uint64')
    labels[0] = model.nr_out-1
    finish_update(labels)
