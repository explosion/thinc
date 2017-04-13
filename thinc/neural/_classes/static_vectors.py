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


def get_word_ids(ops, pad=1, token_drop=0., ignore=None):
    def get_word_ids(docs, drop=0.):
        '''Get word forms.'''
        seqs = []
        ops = Model.ops
        for doc in docs:
            if ignore is not None:
                doc = [token for token in doc if not ignore(token)]
            arr = ops.allocate((len(doc)+pad,), dtype='uint64')
            mask = ops.get_dropout_mask(arr.shape, token_drop)
            for i, token in enumerate(doc):
                if mask is not None and not mask[i]:
                    arr[i] = token.tag
                else:
                    arr[i] = token.lex_id or token.orth
            for i in range(len(doc), len(doc)+pad):
                arr[i] = 0
            seqs.append(ops.asarray(arr))
        return seqs, None
    return layerize(get_word_ids)


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
class StaticVectors(Model):
    '''Load a static embedding table, and learn a linear projection from it.

    Out-of-vocabulary items are modded into the table, receiving an arbitrary
    vector (but the same word will always receive the same vector).
    '''
    name = 'static-vectors'
    def __init__(self, lang, nO):
        Model.__init__(self)
        self.nO = nO
        # This doesn't seem the cleverest solution,
        # but it ensures multiple models load the
        # same copy of spaCy if they're deserialised.
        self.lang = lang
        vectors = self.get_vectors()
        self.nM = vectors.shape[1]
        if self.nM == 0:
            raise ValueError(
                "Cannot create vectors table with dimension 0.\n"
                "If you're using pre-trained vectors, are the vectors loaded?")
        self.nV = vectors.shape[0]

    def get_vectors(self):
        return get_vectors(self.ops, self.lang)

    def begin_update(self, ids, drop=0.):
        vector_table = self.get_vectors()
        vectors = vector_table[ids % vector_table.shape[0]]
        def finish_update(gradients, sgd=None):
            self.d_W += self.ops.batch_outer(gradients, vectors)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return None
        dotted = self.ops.batch_dot(vectors, self.W)
        return dotted, finish_update
