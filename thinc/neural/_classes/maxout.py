from .affine import Affine


class Maxout(Affine):
    name = 'maxout'
    
    @property
    def describe_params(self):
        yield 'W', (self.nr_out, self.nr_in), self.ops.xavier_uniform_init
        yield 'b', (self.nr_out,), None

    def setup(self, nr_out=None, nr_in=None, nr_piece=None, **kwargs):
        self.nr_out = nr_out
        self.nr_in = nr_in
        self.nr_piece = nr_piece
        self.shape = (self.nr_out, self.nr_piece, self.nr_in)
        self.params = Params(self.ops)

    def predict_batch(self, input_bi):
        acts_bop = self.ops.xp.tensordot(input_bi, self.params.W,
                        axes=[[1], [-1]])
        acts_bop += self.params.b
        which_bo = self.ops.argmax(acts_bop, axis=-1)
        return self.ops.take_which(acts_bop, which_bo)

    def begin_update(self, input_BI, dropout=0.0):
        self.check_input(input_BI, expect_batch=True)
        W_OCI = self.params.W
        b_OC = self.params.b
        output_BOC = self.ops.xp.tensordot(input_BI, W_OCI, axes=[[1], [-1]])
        output_BOC += b_OC
        which_BO = self.ops.argmax(output_BOC, axis=-1)
        best_BO = self.ops.take_which(output_BOC, which_BO)
        best_BO, bp_dropout = self.ops.dropout(best_BO, dropout, inplace=True)
        finish_update = self._get_finish_update(input_BI, which_BO)
        return best_BO, bp_dropout(finish_update)

    def _get_finish_update(self, acts_BI, which_BO):
        def finish_update(d_acts_BO, optimizer=None, **kwargs):
            d_acts_BOP = self.ops.allocate((d_acts_BO.shape[0], self.nr_out,
                                           self.nr_piece))
            for i in range(self.nr_piece):
                d_acts_BOP[:, :, i] += d_acts_BO * (which_BO == i)
            self.d_b += d_acts_BOP.sum(axis=0)
            self.d_W += self.ops.xp.tensordot(d_acts_BOP, acts_BI,
                                              axes=[[0], [0]])
            # Bop,opi->Bi
            d_acts_BI = self.ops.xp.tensordot(d_acts_BOP, self.W,
                                              axes=[[1,2], [0, 1]])
            return d_acts_BI
        return finish_update
