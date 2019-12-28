import contextlib

from ...backends import CupyOps
from .model import Model
from ... import describe
from ..util import copy_array


def _uniform_init(lo, hi):
    def wrapped(W, ops):
        copy_array(W, ops.xp.random.uniform(lo, hi, W.shape))

    return wrapped


@describe.attributes(
    nM=describe.Dimension("Vector dimensions"),
    nV=describe.Dimension("Number of vectors"),
    vectors=describe.Weights(
        "Embedding table", lambda obj: (obj.nV, obj.nM), _uniform_init(-0.1, 0.1)
    ),
    d_vectors=describe.Gradient("vectors"),
)
class SimpleEmbed(Model):
    name = "simple-embed"

    def __init__(self, nO, nV=None, column=0, **kwargs):
        Model.__init__(self, **kwargs)
        self.column = column
        self.nO = nO
        self.nV = nV

    def predict(self, ids):
        if ids.ndim == 2:
            ids = ids[:, self.column]
        ids = ids.copy()
        ids[ids >= self.nV] = 0
        return self.vectors[ids]

    def begin_update(self, ids):
        if ids.ndim == 2:
            ids = ids[:, self.column]
        ids[ids >= self.nV] = 0
        vectors = self.vectors[ids]

        def finish_update(gradients):
            if hasattr(self.ops.xp, "scatter_add"):
                self.ops.xp.scatter_add(self.d_vectors, ids, gradients)
            else:
                self.ops.xp.add.at(self.d_vectors, ids, gradients)
            return None

        return vectors, finish_update

    @contextlib.contextmanager
    def use_params(self, params):
        backup = None
        weights = self._mem.weights
        if self.id in params:
            param = params[self.id]
            backup = weights.copy()
            weights[:] = param
        yield
        if backup is not None:
            weights[:] = backup


@describe.attributes(
    nM=describe.Dimension("Vector dimensions"),
    nV=describe.Dimension("Number of vectors"),
    nO=describe.Dimension("Size of output"),
    W=describe.Weights(
        "A projection matrix, to change vector dimensionality",
        lambda obj: (obj.nO, obj.nM),
        lambda W, ops: ops.xavier_uniform_init(W),
    ),
    vectors=describe.Weights(
        "Embedding table", lambda obj: (obj.nV, obj.nM), _uniform_init(-0.1, 0.1)
    ),
    d_W=describe.Gradient("W"),
    d_vectors=describe.Gradient("vectors"),
)
class Embed(Model):
    name = "embed"

    def __init__(self, nO, nM=None, nV=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.is_static = kwargs.get("is_static", False)
        self.column = kwargs.get("column", 0)
        self.nO = nO
        self.nM = nM
        self.nV = nV

    def predict(self, ids):
        if ids.ndim == 2:
            ids = ids[:, self.column]
        if len(ids) < 1000 or isinstance(self.ops, CupyOps):
            vectors = self._embed(ids)
            dotted = self.ops.batch_dot(vectors, self.W)
            return dotted
        uniques, positions = self.ops.xp.unique(ids, return_inverse=True)
        vectors = self._embed(uniques)
        dotted_uniq = self.ops.gemm(vectors, self.W, trans2=True)
        output = dotted_uniq[positions]
        return self.ops.xp.ascontiguousarray(output)

    def begin_update(self, ids):
        if ids.ndim == 2:
            ids = ids[:, self.column]
        vectors = self._embed(ids)
        dotted = self.ops.gemm(vectors, self.W, trans2=True)

        def finish_update(gradients):
            self.d_W += self.ops.gemm(gradients, vectors, trans1=True)
            if not self.is_static:
                gradients = self.ops.gemm(gradients, self.W)
                d_vectors = self.d_vectors
                if hasattr(self.ops.xp, "scatter_add"):
                    self.ops.xp.scatter_add(d_vectors, ids % self.nV, gradients)
                else:
                    self.ops.xp.add.at(d_vectors, ids % self.nV, gradients)
            return None

        return dotted, finish_update

    @contextlib.contextmanager
    def use_params(self, params):
        if self.is_static:
            yield
        else:
            backup = None
            weights = self._mem.weights
            if self.id in params:
                param = params[self.id]
                backup = weights.copy()
                weights[:] = param
            yield
            if backup is not None:
                weights[:] = backup

    def _embed(self, ids):
        vectors = self.vectors
        return vectors[ids % self.nV]
