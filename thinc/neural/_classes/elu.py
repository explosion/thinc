# coding: utf8
from __future__ import unicode_literals

from .affine import Affine


class ELU(Affine):
    def predict(self, input__bi):
        output__bo = Affine.predict(self, input__bi)
        self.ops.elu(output__bo, inplace=True)
        return output__bo

    def begin_update(self, input__bi, drop=0.0):
        output__bo, finish_affine = Affine.begin_update(self, input__bi, drop=drop)

        output_copy = self.ops.xp.ascontiguousarray(output__bo, dtype="f")
        self.ops.elu(output_copy, inplace=True)

        def finish_update(gradient, sgd=None):
            gradient = self.ops.xp.ascontiguousarray(gradient, dtype="f")
            self.ops.backprop_elu(gradient, output_copy, inplace=True)
            return finish_affine(gradient, sgd)

        output__bo[:] = output_copy
        output__bo, bp_dropout = self.ops.dropout(output__bo, drop, inplace=True)
        return output__bo, bp_dropout(finish_update)
