from .affine import Affine
from ... import describe
from ...describe import Dimension, Synapses, Biases


class ReLu(Affine):
    def predict(self, input__BI):
        output__BO = Affine.predict(self, input__BI)
        output__BO = self.ops.xp.ascontiguousarray(output__BO, dtype='float32')
        self.ops.relu(output__BO, inplace=True)
        return output__BO

    def begin_update(self, input__BI, drop=0.0):
        output__BO, finish_affine = Affine.begin_update(self, input__BI, drop=0.)
        output_copy = self.ops.xp.ascontiguousarray(output__BO, dtype='float32')
        self.ops.relu(output_copy, inplace=True)
        def finish_update(gradient, sgd=None):
            gradient = self.ops.xp.ascontiguousarray(gradient, dtype='float32')
            self.ops.backprop_relu(gradient, output_copy, inplace=True)
            return finish_affine(gradient, sgd)
        output__BO[:] = output_copy
        output__BO, bp_dropout = self.ops.dropout(output__BO, drop, inplace=True)
        return output__BO, bp_dropout(finish_update)
