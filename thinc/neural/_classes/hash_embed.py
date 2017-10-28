from .model import Model
from .._lsuv import do_lsuv
from ... import describe
from ...describe import Weights, Dimension, Gradient
from ..util import copy_array
import random
import numpy

def LSUVinit(model, X, y=None):
    if model.vectors is not None:
        do_lsuv(model.ops, model.vectors, model, X)
    return X

def _uniform_init(lo, hi):
    def wrapped(W, ops):
        if (W**2).sum() == 0.:
            copy_array(W, ops.xp.random.uniform(lo, hi, W.shape))
    return wrapped


#@describe.on_data(LSUVinit)
@describe.attributes(
    nO=Dimension("Vector dimensions"),
    nV=Dimension("Number of vectors"),
    vectors=Weights("Embedding table",
        lambda obj: (obj.nV, obj.nO),
        _uniform_init(-0.1, 0.1)
    ),
    d_vectors=Gradient("vectors"),
)
class HashEmbed(Model):
    name = 'hash-embed'
    def __init__(self, nO, nV, seed=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.column = kwargs.get('column', 0)
        self.nO = nO
        self.nV = nV
        self.seed = self.id

    def predict(self, ids):
        if ids.ndim >= 2:
            ids = self.ops.xp.ascontiguousarray(ids[:, self.column], dtype='uint64')
        keys = self.ops.hash(ids, self.seed) % self.nV
        vectors = self.vectors[keys]
        summed = vectors.sum(axis=1)
        return summed

    def begin_update(self, ids, drop=0.):
        if ids.ndim >= 2:
            ids = self.ops.xp.ascontiguousarray(ids[:, self.column], dtype='uint64')
        keys = self.ops.hash(ids, self.seed) % self.nV
        vectors = self.vectors[keys].sum(axis=1)
        mask = self.ops.get_dropout_mask((vectors.shape[1],), drop)
        if mask is not None:
            vectors *= mask
        def finish_update(delta, sgd=None):
            if mask is not None:
                delta *= mask
            keys = self.ops.hash(ids, self.seed) % self.nV
            d_vectors = self.d_vectors
            keys = self.ops.xp.ascontiguousarray(keys.T, dtype='i')
            for i in range(keys.shape[0]):
                self.ops.scatter_add(d_vectors, keys[i], delta)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return None
        return vectors, finish_update
