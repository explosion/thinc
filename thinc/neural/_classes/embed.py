# coding: utf8
from __future__ import unicode_literals

import contextlib

from ..ops import CupyOps
from .model import Model
from ... import describe
from ... import check
from ...check import is_int
from ...describe import Dimension, Weights, Synapses, Gradient
from .._lsuv import do_lsuv
from ..util import copy_array


def _set_dimensions_if_needed(model, X, y=None):
    if model.nV is None:
        max_id = int(X.max()) + 1
        if max_id >= 10000000:  # pragma: no cover
            raise ValueError("TODO error --- really want us to make 1m vectors?")
        model.nV = max_id


def _uniform_init(lo, hi):
    def wrapped(W, ops):
        copy_array(W, ops.xp.random.uniform(lo, hi, W.shape))

    return wrapped


def LSUVinit(model, X, y=None):
    if model.vectors is not None and model.W is not None:
        do_lsuv(model.ops, model.W, model, X)
    return X


@describe.on_data(LSUVinit)
@describe.attributes(
    nM=Dimension("Vector dimensions"),
    nV=Dimension("Number of vectors"),
    nO=Dimension("Size of output"),
    W=Synapses(
        "A projection matrix, to change vector dimensionality",
        lambda obj: (obj.nO, obj.nM),
        lambda W, ops: ops.xavier_uniform_init(W),
    ),
    vectors=Weights(
        "Embedding table", lambda obj: (obj.nV, obj.nM), _uniform_init(-0.1, 0.1)
    ),
    d_W=Gradient("W"),
    d_vectors=Gradient("vectors"),
)
class Embed(Model):
    name = "embed"

    # @property
    # def input_shape(self):
    #    return (self.nB,)

    # @property
    # def output_shape(self):
    #    return (self.nB, self.nO)

    @check.arg(1, is_int)
    def __init__(self, nO, nM=None, nV=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.is_static = kwargs.get("is_static", False)
        self.column = kwargs.get("column", 0)
        self.nO = nO
        self.nM = nM
        self.nV = nV

    # @check.arg(1, is_int_array)
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

    def begin_update(self, ids, drop=0.0):
        if ids.ndim == 2:
            ids = ids[:, self.column]
        mask = self.ops.get_dropout_mask(ids.shape[0], drop)
        if mask is not None:
            ids = ids * (mask > 0)
        vectors = self._embed(ids)
        dotted = self.ops.gemm(vectors, self.W, trans2=True)

        def finish_update(gradients, sgd=None):
            self.d_W += self.ops.gemm(gradients, vectors, trans1=True)
            if not self.is_static:
                gradients = self.ops.gemm(gradients, self.W)
                d_vectors = self.d_vectors
                if hasattr(self.ops.xp, "scatter_add"):
                    self.ops.xp.scatter_add(d_vectors, ids % self.nV, gradients)
                else:
                    self.ops.xp.add.at(d_vectors, ids % self.nV, gradients)
            if sgd is not None:
                if self.is_static:
                    sgd(self.W.ravel(), self.d_W.ravel(), key=self.id)
                else:
                    sgd(self._mem.weights, self._mem.gradient, key=self.id)
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
