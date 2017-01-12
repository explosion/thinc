from .model import Model
from ..exceptions import ShapeError


@describe.input(("B", "I"))
@describe.output(("B", "O"))
@describe.on_data(
    lambda self, X, y: {"B": X.shape[0], "I": X.shape[1], "O": y.max()})
@describe.on_init(lambda self, *args, **kwargs: {"I": args[1], "O": args[0]})
@describe.attributes(
    B=Dimension("Batch size"),
    I=Dimension("Input size"),
    O=Dimension("Output size"),
    W=Synapses("Weights matrix", ("O", "I"), xavier_init),
    B=Biases("Bias vector", ("O",))
)
class Affine(Model):
    '''Computes the linear transform Y = (W @ X) + b.'''
    def predict(self, input__BI):
        return self.ops.affine(self.w.W, self.w.b, input__BI)

    def begin_update(self, input_BI):
        output_BO = self.predict(input_BI)
        def finish_update(grad__BO):
            self.d.W += self.ops.batch_outer(grad__BO, input__BI)
            self.d.b += grad__BO.sum(axis=0)
            return self.ops.batch_dot(grad__BO, self.w.W.T)
        return output__BO, finish_update
