from .affine import Affine


class ReLu(Affine):
    name = 'relu'
    @property
    def describe_params(self):
        '''
        Yields (name, shape, initializer) triples describing the weights directly
        owned by the layer.
        '''
        #def init(W, **kwargs):
        #    W += self.ops.he_normal_init(W.shape, W.shape[-1])
        init = self.ops.xavier_uniform_init
        yield 'W-%s' % self.name, (self.nr_out, self.nr_in), init
        yield 'b-%s' % self.name, (self.nr_out,), None
    
    def predict_batch(self, X):
        output = Affine.predict_batch(self, X)
        return self.ops.clip_low(output, 0, inplace=True)

    def begin_update(self, input_BI, dropout=0.0):
        output_BO, finish_affine = Affine.begin_update(self, input_BI)
        def finish_update(gradient, *args, **kwargs):
            return finish_affine(gradient * (output_BO > 0), *args, **kwargs)
        output_BO = self.ops.clip_low(output_BO, 0, inplace=True)
        return output_BO, finish_update
