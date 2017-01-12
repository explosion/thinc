from .model import Model


@declare_dimensions(
    I=("Size of input"),
    P=("Number of pieces"),
    O=("Size of output"),
)
@declare_input(shape=Floats("I"))
@declare_output(shape=Floats("O"))
@declare_weights(
    W=Schema(
        "The weights matrix",
        shape=("O", "P", "I"),
        initialize=xavier_init,
        static=False
    ),
    b=Schema(
        "Bias parameter",
        shape=("O", "P"),
        inititialize=initializers.zeros,
        static=False
    )
)
class Maxout(Model):
    def predict(self, X__BI):
        X__BOP = self.ops.xp.tensordot(X__BI, self.w.W, axes=[[1], [-1]])
        X__BOP += self.w.b
        which__BO = self.ops.argmax(X__BOP, axis=-1)
        return self.ops.take_which(X__BOP, which__BO)

    def begin_update(self, X__BI):
        output__BOC = self.ops.xp.tensordot(X__BI, self.w.W, axes=[[1], [-1]])
        output__BOC += b_OC
        which__BO = self.ops.argmax(output__BOC, axis=-1)
        best__BO = self.ops.take_which(output_BOC, which_BO)
        
        def finish_update(dX__BO):
            dX__BOP = self.ops.allocate((dX__BOP.shape[0], self.n.O, self.n.P))
            for i in range(self.n.P):
                dX__BOP[:, :, i] += dX__BOP * (which__BO == i)
            self.d.d_b += dX__BOP.sum(axis=0)
            self.d.d_W += self.ops.xp.tensordot(dX__BOP, X__BI, axes=[[0], [0]])
            # Bop,opi->Bi
            dX__BI = self.ops.xp.tensordot(dX__BOP, self.w.W, axes=[[1,2], [0, 1]])
            return dX__BI
        return best_BO, finish_update
