# coding: utf8
from __future__ import unicode_literals

from numpy import prod

from .. import check
from ..check import is_shape


class Memory(object):
    def __init__(self, ops, size=128):
        if size < 0:
            raise ValueError("TODO error re negative size %d" % size)
        self.ops = ops
        self._mem = self.ops.allocate((2, size))
        self._offsets = {}
        self._sizes = {}
        self._i = 0

    @property
    def weights(self):
        return self._mem[0, : self._i]

    @property
    def gradient(self):
        return self._mem[1, : self._i]

    def __contains__(self, name):
        return name in self._offsets

    def __getitem__(self, name):
        offset, col, shape = self._offsets[name]
        size = self._sizes[name]
        return self._mem[col, offset : offset + size].reshape(shape)

    def get(self, name, default=None):
        return self[name] if name in self._offsets else default

    def set(self, value):
        self._mem[0, : self._i] = value

    @check.arg(2, is_shape)
    def add(self, name, shape):
        assert name not in self._offsets, "TODO error"
        self._offsets[name] = (self._i, 0, shape)
        size = prod(shape)
        self._sizes[name] = size
        blob = self._get_blob(size)
        return blob[0].reshape(shape)

    def add_gradient(self, grad_name, param_name):
        assert grad_name not in self._offsets, "TODO error"
        offset, _, shape = self._offsets[param_name]
        size = self._sizes[param_name]
        self._offsets[grad_name] = (offset, 1, shape)
        self._sizes[grad_name] = size
        return self._mem[1, offset : offset + size].reshape(shape)

    def _get_blob(self, nr_req):
        nr_avail = self._mem.shape[1] - (self._i + 1)
        if nr_avail < nr_req:
            self._realloc(max(self._mem.shape[1], nr_req) * 2)
        blob = self._mem[:, self._i : self._i + nr_req]
        self._i += nr_req
        return blob

    def _realloc(self, new_size):
        new_mem = self.ops.allocate((self._mem.shape[0], new_size))
        new_mem[:, : self._i + 1] = self._mem[:, : self._i + 1]
        self._mem = new_mem


#
#    def merge_params(self, others):
#        others = list(others)
#        if not others:
#            return None
#        if not all(other.allow_resize for other in others):
#            raise ValueError("TODO Error")
#        sizes = [other._i+1 for other in others]
#        nr_req = self._i + sum(sizes)
#        if self._mem.shape[1] < nr_req:
#            self._realloc(nr_req)
#        self.allow_resize = False
#        for other in others:
#            other.replace_mem(self._get_blob(other._i))
#
#    def replace_mem(self, mem):
#        if not self.allow_resize:
#            raise ValueError("TODO Error")
#        self.allow_resize = False
#        mem[:] = self._mem[:, :self._i]
#        self._mem = mem
#
#
