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

    def begin_update(self, X__BI):
        output__BOC = self.ops.xp.tensordot(X__BI, self.W, axes=[[1], [-1]])
        output__BOC += b_OC
        which__BO = self.ops.argmax(output__BOC, axis=-1)
        best__BO = self.ops.take_which(output_BOC, which_BO)
        
        def finish_update(dX__BO):
            dX__BOP = self.ops.allocate((dX__BOP.shape[0], self.nO, self.nP))
            for i in range(self.nP):
                dX__BOP[:, :, i] += dX__BOP * (which__BO == i)
            self.d_b += dX__BOP.sum(axis=0)
            self.d_W += self.ops.xp.tensordot(dX__BOP, X__BI, axes=[[0], [0]])
            # Bop,opi->Bi
            dX__BI = self.ops.xp.tensordot(dX__BOP, self.W, axes=[[1,2], [0, 1]])
            return dX__BI
        return best_BO, finish_update
