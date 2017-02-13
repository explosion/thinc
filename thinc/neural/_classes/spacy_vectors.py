import numpy

from ...import describe
from ...describe import Dimension, Synapses, Gradient
from .._lsuv import LSUVinit
from ..ops import NumpyOps
from ...api import layerize
from .model import Model
from ...extra.load_nlp import get_vectors

try:
    import cupy
except ImportError:
    cupy = None


@layerize
def get_word_ids(docs, drop=0.):
    '''Get word forms.'''
    seqs = []
    ops = Model.ops
    for doc in docs:
        arr = numpy.zeros((len(doc)+1,), dtype='uint64')
        for token in doc:
            arr[token.i] = token.lex_id
        arr[len(doc)] = 0
        seqs.append(ops.asarray(arr))
    return seqs, None



@describe.on_data(LSUVinit)
@describe.attributes(
        nM=Dimension("Vector dimensions"),
        nO=Dimension("Size of output"),
        W=Synapses(
            "A projection matrix, to change vector dimensionality",
            lambda obj: (obj.nO, obj.nM),
            lambda W, ops: ops.xavier_uniform_init(W)),
        d_W=Gradient("W"),
)
class SpacyVectors(Model):
    name = 'spacy-vectors'
    def __init__(self, lang, nO):
        Model.__init__(self)
        self.nO = nO
        # This doesn't seem the cleverest solution,
        # but it ensures multiple models load the
        # same copy of spaCy if they're deserialised.
        vectors = get_vectors(self.ops, lang)
        self.lang = lang
        self.nM = vectors.shape[1]
        self.nV = vectors.shape[0]

    def begin_update(self, ids, drop=0.):
        vector_table = get_vectors(self.ops, self.lang)
        ids *= ids < vector_table.shape[0]
        vectors = vector_table[ids]
        def finish_update(gradients, sgd=None):
            self.d_W += self.ops.batch_outer(gradients, vectors)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return None
        dotted = self.ops.batch_dot(vectors, self.W)
        return dotted, finish_update
