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
        output_copy = self.ops.xp.ascontiguousarray(output_BO, dtype='f')
        self.ops.relu(output_copy)
        def finish_update(gradient, *args, **kwargs):
            gradient = self.ops.xp.ascontiguousarray(gradient, dtype='f')
            self.ops.backprop_relu(gradient, output_copy)
            return finish_affine(gradient, *args, **kwargs)
        output_BO[:] = output_copy
        return output_BO, finish_update
