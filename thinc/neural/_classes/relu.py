from .affine import Affine
from ... import describe
from ...describe import Dimension, Synapses, Biases


class ReLu(Affine):
    def predict(self, input__BI):
        output__BO = Affine.predict(self, input__BI)
        self.ops.relu(output__BO, inplace=True)
        return output__BO

    def begin_update(self, input__BI):
        output__BO, finish_affine = Affine.begin_update(self, input__BI)

        output_copy = self.ops.xp.ascontiguousarray(output__BO, dtype='f')
        self.ops.relu(output_copy, inplace=True)
        def finish_update(gradient):
            gradient = self.ops.xp.ascontiguousarray(gradient, dtype='f')
            self.ops.backprop_relu(gradient, output_copy, inplace=True)
            return finish_affine(gradient)
        output__BO[:] = output_copy
        return output__BO, finish_update
