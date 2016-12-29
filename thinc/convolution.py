from .base import Model


class ExtractWindow(Model):
    n = 2
    @property
    def nr_weight(self):
        return 0

    def set_weights(self, *args, **kwargs):
        pass

    def set_gradient(self, *args, **kwargs):
        pass

    def predict_batch(self, X):
        assert self.n == 1
        shape = (X.shape[0], 3, X.shape[1])
        output = self.ops.allocate(shape)

        output[0, 1] = X[0]
        output[0, 2] = X[1]
        for i in range(1, X.shape[0] - 1):
            output[i, 0] = X[i-1]
            output[i, 1] = X[i]
            output[i, 2] = X[i+1]
        output[-1, 0] = X[-2]
        output[-1, 1] = X[-1]
        return output.reshape(output.shape[0], output.shape[1] * output.shape[2])

    #def predict_one(self, X):
    #    shape = (X.shape[0], (self.n*2+1) * X.shape[1])
    #    print(shape)
    #    output = self.ops.allocate(shape)
    #    for i in range(self.n):
    #        # 0 -> (pad, pad, 0, 1, 2) so (0, 2:) = (2 : 5)
    #        # 1 -> (pad, 0, 1, 2, 3) so (1, 1: ) = (1 : 5)
    #        output[i, self.n-i : ] = X[self.n - i : (self.n * 2 + 1)]
    #    for i in range(self.n, X.shape[0] - (self.n + 1)):
    #        output[i] = X[i-self.n : i+1+self.n]
    #    for i in range(X.shape[0] - (self.n + 1), X.shape[0]):
    #        # 5 -> (3, 4, 5, 6, pad) so (5, :4) = (3 : 7)
    #        # 6 -> (4, 5, 6, pad, pad) so (6, :3) = (4 : 7)
    #        output[i, : (self.n * 2 - i)] = X[i - self.n : ]
    #    return output

    def begin_update(self, X, dropout=0.0):
        output = self.predict_batch(X)
        output, bp_dropout = self.ops.dropout(output, dropout)
        finish_update = self._get_finish_update()
        return output, bp_dropout(finish_update)

    def _get_finish_update(self):
        def finish_update(gradient, optimizer=None, **kwargs):
            assert self.n == 1
            shape = (gradient.shape[0], 3, int(gradient.shape[-1] / 3))
            gradient = gradient.reshape(shape)
            output = self.ops.allocate((gradient.shape[0], gradient.shape[-1]))
            # Central column of gradient is the word's own feature
            output += gradient[:, 1, :]
            # First column is word i as left feature of word i+1
            output[:-1, ] += gradient[1:, 0, :]
            return output
            #output[:, 0, :] 
            ## feature_as += feature_has
            ## When rightward: first words *as* fewer features,
            ## last words *have* fewer features
            ## When leftward: last words *as* fewer features,
            ## first words *have* fewer features
            #for i in range(self.n):
            #    # Update for rightward features
            #    # grad[1] += grad[0, 3]
            #    # grad[2] += grad[1, 3] + grad[0, 4]
            #    output[self.n-i : ] += gradient[: -(self.n-i), (self.n * 2-i)]
            #for i in range(self.n):
            #    # Update for leftward features
            #    # i=0: Left-left
            #    # grad[4] += grad[6, 0]
            #    # i=1: Left
            #    # grad[4] += grad[5, 1]
            #    # grad[5] += grad[6, 1]
            #    output[: -(self.n-i)] += gradient[(self.n-i) : , i]
        return finish_update
