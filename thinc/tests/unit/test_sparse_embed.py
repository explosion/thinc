from ...neural._classes.sparse_embed import SparseEmbed
import numpy


def test_create_sparse_embed():
    embed = SparseEmbed(300)


def test_predict_sparse_embed():
    embed = SparseEmbed(300)
    ids = numpy.ones((10,), dtype='uint64')
    vectors = embed.predict(ids)
    assert vectors.shape == (10, 300)
    assert vectors.sum() != 0.


def test_sparse_embed_begin_update():
    embed = SparseEmbed(300)
    ids = numpy.ones((10,), dtype='uint64')
    vectors, finish_update = embed.begin_update(ids)
    assert vectors.shape == (10, 300)
    assert vectors.sum() != 0.
    gradient = embed.ops.allocate(vectors.shape)
    finish_update(gradient)

