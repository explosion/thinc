from .affine import Affine


class ReLu(Affine):
    def predict(self, input__BI):
        output__BO = Affine.predict(self, input__BI)
        output__BO = self.ops.relu(output__BO, inplace=False)
        return output__BO

    def begin_update(self, input__BI, drop=0.0):
        output__BO, finish_affine = Affine.begin_update(self, input__BI, drop=0.0)
        output__BO = self.ops.relu(output__BO)

        def finish_update(gradient):
            gradient = self.ops.backprop_relu(gradient, output__BO)
            return finish_affine(gradient)

        dropped, bp_dropout = self.ops.dropout(output__BO, drop, inplace=False)
        return dropped, bp_dropout(finish_update)
