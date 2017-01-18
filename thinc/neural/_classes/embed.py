from .model import Model
from ... import describe
from ... import check
from ...check import is_int_array
from ... describe import Dimension, Weights, Synapses, Gradient


def _set_dimensions_if_needed(model, X, y=None):
    if model.nV is None:
        max_id = int(X.max()) + 1
        if max_id >= 10000000: # pragma: no cover
            raise ValueError("TODO error --- really want us to make 1m vectors?")
        model.nV = max_id


def _uniform_init(lo, hi):
    def wrapped(W, ops):
        W[:] = ops.xp.random.uniform(lo, hi, W.shape)
    return wrapped


@describe.on_data(_set_dimensions_if_needed)
@describe.attributes(
    nM=Dimension("Vector dimensions"),
    nV=Dimension("Number of vectors"),
    nO=Dimension("Size of output"),
    W=Synapses(
        "A projection matrix, to change vector dimensionality",
        lambda obj: (obj.nO, obj.nM),
        lambda W, ops: ops.xavier_uniform_init(W)),
    vectors=Weights("Embedding table",
        lambda obj: (obj.nV, obj.nM),
        _uniform_init(-0.1, 0.1)
    ),
    d_W=Gradient("W"),
    d_vectors=Gradient("vectors")
)
class Embed(Model):
    name = 'embed'

    #@property
    #def input_shape(self):
    #    return (self.nB,)

    #@property
    #def output_shape(self):
    #    return (self.nB, self.nO)

    def __init__(self, nO, nM=None, nV=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.nM = nM
        self.nV = nV

    @check.arg(1, is_int_array)
    def predict(self, ids):
        if len(ids) < 1000:
            vectors = self._embed(ids)
            return self.ops.batch_dot(vectors, self.W)
        id_map = {}
        for i, id_ in enumerate(ids):
            if id_ not in id_map:
                id_map[id_] = [i]
            else:
                id_map[id_].append(i)
        mapped = sorted(id_map.items())
        vectors = self._embed([id_ for id_, _ in mapped])
        result = self.ops.batch_dot(vectors, self.W)
        output = self.ops.allocate((len(ids), self.nO))
        for i, (_, occurs) in enumerate(mapped):
            for j in occurs:
                output[j] = result[i]
        return output

    def begin_update(self, ids, drop=0.):
        def finish_update(gradients, sgd=None):
            self.d_W += self.ops.batch_outer(gradients, self._embed(ids))
            gradients = self.ops.batch_dot(gradients, self.W.T)
            for id_, delta_in in zip(ids, gradients):
                id_ = int(id_)
                if id_ < self.d_vectors.shape[0]:
                    self.d_vectors[id_] += delta_in
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=id(self._mem))
            return None
        return self.predict(ids), finish_update

    def _embed(self, ids):
        vectors = self.ops.allocate((len(ids), self.nM))
        for i, id_ in enumerate(ids):
            id_ = int(id_)
            if id_ < self.vectors.shape[0]:
                vectors[i] = self.vectors[id_]
        return vectors
