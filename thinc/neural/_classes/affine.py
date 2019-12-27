from .model import Model
from ... import describe


@describe.attributes(
    nI=describe.Dimension("Input size"),
    nO=describe.Dimension("Output size"),
    W=describe.Weights(
        "Weights matrix",
        lambda obj: (obj.nO, obj.nI),
        lambda W, ops: ops.xavier_uniform_init(W),
    ),
    b=describe.Weights("Bias vector", lambda obj: (obj.nO,)),
    d_W=describe.Gradient("W"),
    d_b=describe.Gradient("b"),
)
class Affine(Model):
    """Computes the linear transform Y = (W @ X) + b."""

    name = "affine"

    def __init__(self, nO=None, nI=None, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.nI = nI
        self.drop_factor = kwargs.get("drop_factor", 1.0)

    def predict(self, X):
        Y = self.ops.gemm(X, self.W, trans2=True)
        Y += self.b
        return Y

    def begin_update(self, X):
        Y = self.predict(X)

        def backprop_affine(dY):
            dY = self.ops.xp.ascontiguousarray(dY)
            self.ops.gemm(dY, X, trans1=True, out=self.d_W)
            self.d_b += dY.sum(axis=0)
            return self.ops.gemm(dY, self.W)

        return Y, backprop_affine
