from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases


@describe.output(("nO",))
@describe.input(("nI",))
@describe.attributes(
    nI=Dimension("Size of input"),
    nP=Dimension("Number of pieces"),
    nO=Dimension("Size of output"),
    W=Synapses("The weights matrix", ("nO", "nI")),
    b=Biases("Bias parameter", ("nO", "nI"))
)
class Softmax(Model):
    def predict(self, input__BI):
        output__BO = self.ops.affine(self.W, self.b, input__BI)
        self.ops.softmax(output__BO, inplace=True)
        return output__BO

    def begin_update(self, input__BI):
        output__BO = self.predict(input_BI)
        def finish_update(grad__BO):
            self.d_W += self.ops.batch_outer(grad__BO, input__BI)
            self.d_b += grad__BO.sum(axis=0)
            return self.ops.batch_dot(grad__BO, self.W.T)
        return output__BO, finish_update
