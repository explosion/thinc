import numpy
from thinc.api import HashEmbed, fix_random_seed, remove_random_seed


def test_init():
    model = HashEmbed(64, 1000).initialize()
    assert model.get_dim("nV") == 1000
    assert model.get_dim("nO") == 64
    assert model.get_param("E").shape == (1000, 64)


def test_seed_same_bucket():
    """ The vectors are only the same if the random seed AND the HashEmbed seeds are fixed"""
    fix_random_seed(0)
    model1 = HashEmbed(64, 1000, seed=1).initialize()
    model2 = HashEmbed(64, 1000, seed=1).initialize()
    arr = numpy.ones((1,), dtype="uint64")
    vector1 = model1.predict(arr)
    vector2 = model2.predict(arr)
    assert vector1.sum() == vector2.sum()


def test_seed_same_bucket_v2():
    """ The vectors are the same if the random seed is deduced from the global one"""
    fix_random_seed(0)
    model1 = HashEmbed(64, 1000).initialize()
    model2 = HashEmbed(64, 1000).initialize()
    arr = numpy.ones((1,), dtype="uint64")
    vector1 = model1.predict(arr)
    vector2 = model2.predict(arr)
    assert vector1.sum() == vector2.sum()


def test_seed_not_fixed():
    remove_random_seed()
    model1 = HashEmbed(64, 1000, seed=1).initialize()
    model2 = HashEmbed(64, 1000, seed=1).initialize()
    arr = numpy.ones((1,), dtype="uint64")
    vector1 = model1.predict(arr)
    vector2 = model2.predict(arr)
    assert vector1.sum() != vector2.sum()


def test_seed_changes_bucket():
    """ The vectors are not the same if the HashEmbed's seed is different"""
    fix_random_seed(0)
    model1 = HashEmbed(64, 1000, seed=2).initialize()
    model2 = HashEmbed(64, 1000, seed=1).initialize()
    arr = numpy.ones((1,), dtype="uint64")
    vector1 = model1.predict(arr)
    vector2 = model2.predict(arr)
    assert vector1.sum() != vector2.sum()
