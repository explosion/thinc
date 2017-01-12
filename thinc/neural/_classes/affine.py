from .model import Model
from ..exceptions import ShapeError


@describe.input(("B", "I"))
@describe.output(("B", "O"))
@describe.weights(
    W=("Weights matrix", ("O", "I"), xavier_init),
    b=("Bias",           ("O",),     None)
)
@describe.dimensions(
    B="Batch size",
    I="Input size",
    O="Output size",
    on_init=lambda args: {"I": args.get(1), "O": args.get(0)},
    on_X=lambda X: {"B": X.shape[0], "I": X.shape[1]},
    on_y=lambda y: {"B": y.shape[0], "O": y.max()}
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
