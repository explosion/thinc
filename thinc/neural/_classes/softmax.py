from .affine import Affine
from ... import describe
from ...describe import Dimension, Synapses, Biases


@describe.attributes(
    W=Synapses("Weights matrix",
        lambda obj: (obj.nO, obj.nI),
        lambda W, ops: None)
)
class Softmax(Affine):
    name = 'softmax'
    def predict(self, input__BI):
        output__BO = self.ops.affine(self.W, self.b, input__BI)
        self.ops.softmax(output__BO, inplace=True)
        return output__BO

    def begin_update(self, input__BI, drop=0.):
        output__BO = self.predict(input__BI)
        def finish_update(grad__BO, sgd=None):
            self.d_W += self.ops.batch_outer(grad__BO, input__BI)
            self.d_b += grad__BO.sum(axis=0)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient,
                    key=id(self._mem))
            return self.ops.batch_dot(grad__BO, self.W.T)
        return output__BO, finish_update
