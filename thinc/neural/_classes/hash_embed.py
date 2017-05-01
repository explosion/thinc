from .model import Model
from .embed import _uniform_init
from .._lsuv import do_lsuv
from ... import describe
from ...describe import Weights, Dimension, Gradient
import random
import numpy

def LSUVinit(model, X, y=None):
    if model.vectors is not None:
        do_lsuv(model.ops, model.vectors, model, X)
    return X

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
        vectors = self.vectors[self.ops.hash(ids, self.seed) % self.nV]
        return vectors

    def begin_update(self, ids, drop=0.):
        if ids.ndim >= 2:
            ids = self.ops.xp.ascontiguousarray(ids[:, self.column], dtype='uint64')
        vectors = self.predict(ids)
        mask = self.ops.get_dropout_mask((vectors.shape[1],), drop)
        if mask is not None:
            vectors *= mask
        def finish_update(delta, sgd=None):
            if mask is not None:
                delta *= mask
            keys = self.ops.hash(ids, self.seed) % self.nV
            if hasattr(self.ops.xp, 'scatter_add'):
                self.ops.xp.scatter_add(self.d_vectors, keys, delta)
            else:
                self.ops.xp.add.at(self.d_vectors, keys, delta)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return None
        return vectors, finish_update
