# coding: utf8
from __future__ import unicode_literals

from .affine import Affine
from ... import check
from ...check import has_shape


class ReLu(Affine):
    @check.arg(1, has_shape(("nB", "nI")))
    def predict(self, input__BI):
        output__BO = Affine.predict(self, input__BI)
        output__BO = self.ops.relu(output__BO, inplace=False)
        return output__BO

    @check.arg(1, has_shape(("nB", "nI")))
    def begin_update(self, input__BI, drop=0.0):
        output__BO, finish_affine = Affine.begin_update(self, input__BI, drop=0.0)
        output__BO = self.ops.relu(output__BO)

        @check.arg(0, has_shape(("nB", "nO")))
        def finish_update(gradient, sgd=None):
            gradient = self.ops.backprop_relu(gradient, output__BO)
            return finish_affine(gradient, sgd)

        dropped, bp_dropout = self.ops.dropout(output__BO, drop, inplace=False)
        return dropped, bp_dropout(finish_update)
