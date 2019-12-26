from .affine import Affine


class ReLu(Affine):
    def predict(self, input__BI):
        output__BO = Affine.predict(self, input__BI)
        output__BO = self.ops.relu(output__BO, inplace=False)
        return output__BO

    def begin_update(self, input__BI):
        output__BO, finish_affine = Affine.begin_update(self, input__BI)
        output__BO = self.ops.relu(output__BO)

        def finish_update(gradient):
            gradient = self.ops.backprop_relu(gradient, output__BO)
            return finish_affine(gradient)

        return output__BO, finish_update
