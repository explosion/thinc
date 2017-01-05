from .affine import Affine


class ReLu(Affine):
    name = 'relu'
    
    def predict_batch(self, X):
        output = Affine.predict_batch(self, X)
        return self.ops.clip_low(output, 0, inplace=True)

    def begin_update(self, input_BI, dropout=0.0):
        output_BO, finish_affine = Affine.begin_update(self, input_BI)
        def finish_update(gradient, *args, **kwargs):
            return finish_affine(gradient * (output_BO > 0), *args, **kwargs)
        return output_BO, finish_update
