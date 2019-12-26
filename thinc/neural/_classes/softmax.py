from .affine import Affine
from ... import describe


@describe.attributes(
    W=describe.Weights(
        "Weights matrix", lambda obj: (obj.nO, obj.nI), lambda W, ops: None
    )
)
class Softmax(Affine):
    name = "softmax"

    def predict(self, input__BI):
        output__BO = self.ops.affine(self.W, self.b, input__BI)
        self.ops.softmax(output__BO, inplace=True)
        return output__BO

    def begin_update(self, input__BI, drop=0.0):
        output__BO = self.predict(input__BI)

        def finish_update(grad__BO, sgd=None):
            self.ops.gemm(grad__BO, input__BI, trans1=True, out=self.d_W)
            self.d_b += grad__BO.sum(axis=0)
            grad__BI = self.ops.gemm(grad__BO, self.W)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return grad__BI

        return output__BO, finish_update
