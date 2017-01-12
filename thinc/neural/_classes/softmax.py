from .affine import Affine


@describe.input(shape=("B", "I",), type=Floats())
@describe.output(shape=("B",), type=Ints(), max="O")
@describe.weights(
    W=Floats("Synapses matrix", shape=("O", "I"), initialize=None),
    b=Floats("Bias vector", shape=("O",), inititialize=None)
)
@describe.dimensions(
    {"B": "Batch size", "I": "Input size", "O": "Number of classes"},
    from_init=lambda args: {"I": args.get(1), "O": args.get(0)},
    from_X=lambda X: {"B": X.shape[0], "I": X.shape[1]},
    from_y=lambda y: {"O": y.max()})
class SoftmaxArgmax(Model):
    def predict_batch(self, input__BI):
        output__BO = self.ops.affine(self.w.W, self.w.b, input__BI)
        self.ops.softmax(output__BO, inplace=True)
        return self.ops.argmax(output__BO)

    def begin_update(self, input__BI):
        output__BO = self.predict(input_BI)
        def finish_update(grad__BO):
            self.d_W += self.ops.batch_outer(grad__BO, input__BI)
            self.d_b += grad__BO.sum(axis=0)
            return self.ops.batch_dot(grad__BO, self.w.W.T)
        return output__BO, finish_update
