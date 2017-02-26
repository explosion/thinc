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
        self.nO = nO
        self.nV = nV
        print("Hash embed", self.nV, self.nO)
        self.word_weights = {}
        self.d_word_weights = {}
        self.seed = id(self)

    def predict(self, ids):
        vectors = self.vectors[self.ops.hash(ids, self.seed) % self.nV]
        for i, id_ in enumerate(ids):
            vectors[i] *= self.word_weights.get(id_, 0.0001)
        return vectors

    def begin_update(self, ids, drop=0.):
        def finish_update(delta, sgd=None):
            word_weights = self.ops.allocate((delta.shape[0],))
            for i, id_ in enumerate(ids):
                word_weights[i] = self.word_weights.get(id_, 0.)
                if word_weights[i] != 0.:
                    self.d_word_weights.setdefault(id_, self.ops.allocate(1))
                    self.d_word_weights[id_] += delta[i].sum()
                else:
                    self.word_weights[id_] = random.random()
                    self.word_weights.setdefault(id_,
                        numpy.random.uniform(-1., 1., 1))

            keys = self.ops.hash(ids, self.seed) % self.nV
            #self.ops.xp.add.at(self.d_vectors, keys, delta)
            self.ops.xp.add.at(self.d_vectors, keys, word_weights.dot(delta))

            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=id(self._mem))
                for id_, gradient in self.d_word_weights.items():
                    sgd(self.word_weights[id_], gradient, key=id_)
                self.d_word_weights = {}
            return None
        return self.predict(ids), finish_update


