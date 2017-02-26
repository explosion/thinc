from .model import Model
from .embed import _uniform_init
from ... import describe
from ...describe import Weights, Dimension, Gradient
import random
import numpy


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
    def __init__(self, nO, nV, seed=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.column = kwargs.get('column', 0)
        self.nO = nO
        self.nV = nV
        self.seed = self.id

    def predict(self, ids):
        if ids.ndim == 2:
            ids = ids[:, self.column]
        vectors = self.vectors[self.ops.hash(ids, self.seed) % self.nV]
        return vectors

    def begin_update(self, ids, drop=0.):
        if ids.ndim == 2:
            ids = ids[:, self.column]
        def finish_update(delta, sgd=None):
            keys = self.ops.hash(ids, self.seed) % self.nV
            self.ops.xp.add.at(self.d_vectors, keys, delta)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return None
        return self.predict(ids), finish_update


