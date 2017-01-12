from .relu import ReLu
from .affine import Affine


class ELU(Model):
    def predict_batch(self, X):
        output = self.ops.xp.ascontiguousarray(
                    Affine.predict_batch(self, X), dtype='f')
        self.ops.elu(output, inplace=True)
        return output

    def begin_update(self, input_BI, dropout=0.0):
        output_BO, finish_affine = Affine.begin_update(self, input_BI)

        output_copy = self.ops.xp.ascontiguousarray(output_BO, dtype='f')
        self.ops.elu(output_copy, inplace=True)
        def finish_update(gradient, *args, **kwargs):
            gradient = self.ops.xp.ascontiguousarray(gradient, dtype='f')
            self.ops.backprop_elu(gradient, output_copy, inplace=True)
            return finish_affine(gradient, *args, **kwargs)
        output_BO[:] = output_copy
        return output_BO, finish_update
