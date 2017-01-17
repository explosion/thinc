from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases


@describe.output(("nO",))
@describe.input(("nI",))
@describe.attributes(
    nI=Dimension("Size of input"),
    nP=Dimension("Number of pieces"),
    nO=Dimension("Size of output"),
    W=Synapses("The weights matrix", ("nO", "nP", "nI"),
        lambda W, ops: ops.xavier_uniform_init),
    b=Biases("Bias parameter", ("nO", "nP"))
)
class Maxout(Model): # pragma: no cover
    def predict(self, X__BI):
        X__BOP = self.ops.xp.tensordot(X__BI, self.w.W, axes=[[1], [-1]])
        X__BOP += self.b
        which__BO = self.ops.argmax(X__BOP, axis=-1)
        return self.ops.take_which(X__BOP, which__BO)

    def begin_update(self, X__bi):
        output__boc = self.ops.xp.tensordot(X__bi, self.W, axes=[[1], [-1]])
        output__boc += self.b
        which__bo = self.ops.argmax(output__boc, axis=-1)
        best__bo = self.ops.take_which(output__boc, which__bo)
        
        def finish_update(dX__bo):
            dX__bop = self.ops.backprop_take(dX__bo, which__bo, self.nP)
            self.d_b += dX__bop.sum(axis=0)
            self.d_W += self.ops.xp.tensordot(dX__bop, X__bi, axes=[[0], [0]])
            # Bop,opi->Bi
            dX__bi = self.ops.xp.tensordot(dX__BOP, self.W, axes=[[1,2], [0, 1]])
            return dX__bi
        return best__bo, finish_update
