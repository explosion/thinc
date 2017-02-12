import numpy

from ...import describe
from ...describe import Dimension, Synapses, Gradient
from .._lsuv import LSUVinit
from ..ops import NumpyOps
from ...api import layerize
from .model import Model

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
        arr = ops.xp.zeros((len(doc)+1,), dtype='uint64')
        for token in doc:
            arr[token.i] = token.orth
        arr[len(doc)] = 0
        seqs.append(arr)
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
    ops = NumpyOps()
    name = 'spacy-vectors'
    def __init__(self, nlp, nO):
        Model.__init__(self)
        self._id_map = {0: 0}
        self.nO = nO
        self.nM = nlp.vocab.vectors_length
        self.nlp = nlp

    @property
    def nV(self):
        return len(self.nlp.vocab)

    def begin_update(self, ids, drop=0.):
        if not isinstance(ids, numpy.ndarray):
            ids = ids.get()
            gpu_in = True
        else:
            gpu_in = False
        uniqs, inverse = numpy.unique(ids, return_inverse=True)
        vectors = self.ops.allocate((uniqs.shape[0], self.nM))
        for i, orth in enumerate(uniqs):
            lex = self.nlp.vocab[orth]
            if lex.vector_norm:
                vectors[i] = lex.vector / lex.vector_norm
        def finish_update(gradients, sgd=None):
            if gpu_in:
                gradients = gradients.get()
            self.d_W += self.ops.batch_outer(gradients, vectors[inverse, ])
            if sgd is not None:
                ops = sgd.ops
                sgd.ops = self.ops
                sgd(self._mem.weights, self._mem.gradient, key=id(self._mem))
                sgd.ops = ops
            return None
        dotted = self.ops.batch_dot(vectors, self.W)
        if gpu_in:
            return cupy.asarray(dotted[inverse, ]), finish_update
        else:
            return dotted[inverse, ], finish_update
