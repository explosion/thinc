import pytest
import numpy

from ...neural._classes.hash_embed import HashEmbed


def test_init():
    model = HashEmbed(64, 1000)
    assert model.nV == 1000
    assert model.nO == 64
    assert model.vectors.shape == (1000, 64)
    #assert model.word_weights.shape == (1000,)


def test_seed_changes_bucket():
    model1 = HashEmbed(64, 1000, seed=2)
    model2 = HashEmbed(64, 1000, seed=1)
    arr = numpy.ones((1,), dtype='uint64')
    vector1 = model1(arr)
    vector2 = model2(arr)
    assert vector1.sum() != vector2.sum()


