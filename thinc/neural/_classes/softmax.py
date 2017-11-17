from .affine import Affine
from ... import describe
from ...describe import Dimension, Synapses, Biases
from ...check import has_shape
from ... import check


@describe.attributes(
    W=Synapses("Weights matrix",
        lambda obj: (obj.nO, obj.nI),
        lambda W, ops: None)
)
class Softmax(Affine):
    name = 'softmax'
    def predict(self, input__BI):
        output__BO = self.ops.gemm(input__BI, self.W, trans2=True)
        output__BO += self.b
        output__BO = self.ops.softmax(output__BO, inplace=False)
        return output__BO

    def begin_update(self, input__BI, drop=0.):
        output__BO = self.predict(input__BI)
        def finish_update(grad__BO, sgd=None):
            self.ops.add_batch_outer(self.d_W, grad__BO, input__BI)
            self.d_b += grad__BO.sum(axis=0)
            grad__BI = self.ops.dot(grad__BO, self.W)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return grad__BI
        return output__BO, finish_update
