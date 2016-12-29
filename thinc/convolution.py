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
        nr_feat = self.n * 2 + 1
        shape = (X.shape[0], nr_feat) + X.shape[1:]
        output = self.ops.allocate(shape)
        # Let L(w[i]) LL(w[i]), R(w[i]), RR(w[i])
        # denote the words one and two to the left and right of word i.
        # So L(w[i]) == w[i-1] for i >= 1 and {empty} otherwise.
        # output[i, n-1] will be L(w[i]) so w[i-1] or empty
        # output[i, n-2] will be LL(w[i]) so w[i-2] or empty
        # etc
        for i in range(self.n):
            output[i+1:, i] = X[:-(i+1)]
        # output[i, n] will be w[i]
        output[:, self.n] = X
        # Now R(w[i]), RR(w[i]) etc
        for i in range(1, self.n+1):
            output[:-i, self.n + i] = X[:-i]
        return output.reshape(shape[0], self.ops.xp.prod(shape[1:]))

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
            # Word w[i+1] is the R feature of w[i]. So
            # grad(w[i+1]) += grad(R[i, n+1])
            # output[:-1] += gradient[1:, self.n+1] 
            for i in range(1, self.n+1): # As rightward features
                output[:-i] += gradient[i:, self.n+i]
            # Word w[i-1] is the L feature of w[i]. So
            # grad(w[i-1]) += grad(L[i, n-1])
            # output[1:] += gradient[:-1, self.n-1] 
            for i in range(1, self.n+1): # As leftward features 
                output[i:] += gradient[:-i, self.n-i]
            # Central column of gradient is the word's own feature
            output += gradient[:, self.n, :]
            return output
        return finish_update
