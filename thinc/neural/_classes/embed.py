import numpy as np
from preshed.maps import PreshMap
import contextlib
from .model import Model
from ... import describe
from ... import check
from ...check import is_int_array, is_int
from ... describe import Dimension, Weights, Synapses, Gradient
from .._lsuv import svd_orthonormal, do_lsuv


def _set_dimensions_if_needed(model, X, y=None):
    if model.nV == None:
        max_id = int(X.max()) + 1
        if max_id >= 10000000: # pragma: no cover
            raise ValueError("TODO error --- really want us to make 1m vectors?")
        model.nV = max_id


def _uniform_init(lo, hi):
    def wrapped(W, ops):
        W[:] = ops.xp.random.uniform(lo, hi, W.shape)
    return wrapped


def LSUVinit(model, X, y=None):
    if model.vectors is not None and model.W is not None:
        do_lsuv(model.ops, model.vectors, model, X)
        do_lsuv(model.ops, model.W, model, X)
    return X


@describe.on_data(_set_dimensions_if_needed, LSUVinit)
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

    @check.arg(1, is_int)
    def __init__(self, nO, nM=None, nV=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.is_static = kwargs.get('is_static', False)
        self.nO = nO
        self.nM = nM
        self.nV = nV
        self._id_map = PreshMap()
        self._id_map[0] = 0

    #@check.arg(1, is_int_array)
    def predict(self, ids):
        ids = self._remap_ids(ids)
        if len(ids) < 1000:
            vectors = self._embed(ids)
            dotted = self.ops.batch_dot(vectors, self.W)
            return dotted
        uniques, positions = self._unique_ids(ids)
        vectors = self._embed(uniques)
        dotted_uniq = self.ops.batch_dot(vectors, self.W)
        output = self.ops.allocate((len(ids), self.nO))
        for i, id_ in enumerate(uniques):
            for j in positions[id_]:
                output[j] = dotted_uniq[i]
        return output

    def begin_update(self, orig_ids, drop=0.):
        def finish_update(gradients, sgd=None):
            ids = self._remap_ids(orig_ids)
            vectors = self._embed(ids)
            self.d_W += self.ops.batch_outer(gradients, vectors)
            if not self.is_static:
                gradients = self.ops.batch_dot(gradients, self.W.T)
                d_vectors = self.d_vectors
                n_vector = d_vectors.shape[0]
                for id_, delta_in in zip(ids, gradients):
                    if id_ < 0:
                        continue
                    if id_ < n_vector:
                        d_vectors[id_] += delta_in
            if sgd is not None:
                if self.is_static:
                    sgd(self.W.flatten(), self.d_W.flatten(), key=id(self._mem))
                else:
                    sgd(self._mem.weights, self._mem.gradient, key=id(self._mem))
            return None
        return self.predict(orig_ids), finish_update

    @contextlib.contextmanager
    def use_params(self, params):
        if self.is_static:
            yield
        else:
            backup = None
            weights = self._mem.weights
            if id(self._mem) in params:
                param = params[id(self._mem)]
                backup = weights.copy()
                weights[:] = param
            yield
            if backup is not None:
                weights[:] = backup

    def _embed(self, ids):
        return self.vectors[ids * (ids < self.vectors.shape[0])]

    def _remap_ids(self, ids):
        return self.ops.remap_ids(self._id_map, ids, len(self._id_map))

    def _unique_ids(self, ids):
        id_map = {}
        for i, id_ in enumerate(ids.flatten()):
            if id_ not in id_map:
                id_map[id_] = [i]
            else:
                id_map[id_].append(i)
        uniques = self.ops.asarray(sorted(id_map.keys()), dtype='uint64')
        return uniques, id_map
