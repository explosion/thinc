from ... import describe
from .model import Model
from ...describe import Dimension, Synapses, Biases, Gradient


@describe.attributes(
    nI=Dimension("Input size"),
    nO=Dimension("Output size"),
    W=Synapses(
        "Weights matrix",
        lambda obj: (obj.nO, obj.nI),
        lambda W, ops: ops.xavier_uniform_init(W),
    ),
    b=Biases("Bias vector", lambda obj: (obj.nO,)),
    d_W=Gradient("W"),
    d_b=Gradient("b"),
)
class Mish(Model):
    """Dense layer with mish activation.

    https://arxiv.org/pdf/1908.08681.pdf
    """

    name = "mish"

    def __init__(self, nO=None, nI=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.nI = nI
        self.drop_factor = kwargs.get("drop_factor", 1.0)

    def predict(self, X):
        Y = self.ops.affine(self.W, self.b, X)
        Y = self.ops.mish(Y)
        return Y

    def begin_update(self, X, drop=0.0):
        if drop is None:
            return self.predict(X), None
        Y1 = self.ops.affine(self.W, self.b, X)
        Y2 = self.ops.mish(Y1)
        drop *= self.drop_factor
        Y3, bp_dropout = self.ops.dropout(Y2, drop)

        def finish_update(dY2, sgd=None):
            dY1 = self.ops.backprop_mish(dY2, Y1)
            self.ops.gemm(dY1, X, trans1=True, out=self.d_W)
            self.d_b += dY1.sum(axis=0)
            dX = self.ops.gemm(dY1, self.W)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return dX

        return Y3, bp_dropout(finish_update)
