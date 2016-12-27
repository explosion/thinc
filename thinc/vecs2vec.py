from .base import Model


class MeanPooling(Model):
    def predict_batch(self, X):
        means = []
        for x in X:
            means.append(self.ops.mean(x, axis=0))
        return self.ops.asarray(means)

    def begin_update(self, X, dropout=0.0):
        mask = self.ops.get_dropout(X.shape, drop)
        X *= mask

        def finish_update(gradient):
            batch_grads = []
            for i, x in enumerate(X):
                grad_i = self.ops.allocate(x.shape)
                if x.shape[0] != 0:
                    grad_i += gradient[i] / x.shape[0]
                grad_i *= mask[i]
                batch_grads.append(grad_i)
            return batch_grads
        return self.predict_batch(X), finish_update


class MaxPooling(Model):
    def predict_batch(self, X):
        means = []
        for x in X:
            means.append(self.ops.max(x, axis=0))
        return self.ops.asarray(means)

    def predict_batch(self, X):
        maxes = []
        for x in X:
            if x.shape[0] == 0:
                maxes.append(self.ops.allocate(x.shape[1:], dtype='f'))
            else:
                maxes.append(x.max(axis=0))
        return self.ops.asarray(maxes)

    def begin_update(self, X, dropout=0.0):
        mask = dropout(X, drop)
        def finish_update(gradient):
            mask = dropout(X, drop, inplace=True)
            batch_grads = []
            for i, x in enumerate(X):
                grad_i = self.ops.allocate(x.shape)
                if x.shape[0] != 0:
                    grad_i += gradient[i] * (x == x.max(axis=0))
                grad_i *= mask[i]
                batch_grads.append(grad_i)
            return batch_grads
        return self.predict_batch(X), finish_update


class MinPooling(Model):
    def predict_batch(self, X):
        maxes = []
        for x in X:
            if x.shape[0] == 0:
                maxes.append(self.ops.allocate(x.shape[1:]))
            else:
                maxes.append(x.min(axis=0))
        return self.ops.asarray(maxes)

    def begin_update(self, X, dropout=0.0):
        mask = dropout(X, drop, inplace=True)
        def finish_update(gradient, optimizer, L2=0.0):
            batch_grads = []
            for i, x in enumerate(X):
                grad_i = self.ops.allocate(x.shape)
                if x.shape[0] != 0:
                    grad_i += gradient[i] * (x == x.min(axis=0))
                grad_i *= mask[i]
                batch_grads.append(grad_i)
            return batch_grads
        return self.predict_batch(X), finish_update


class MultiPooling(NeuralNet):
    def predict_batch(self, X):
        return self.ops.concatenate([in_.predict_batch(X) for in_ in self.inputs], axis=1)
 
    def begin_update(self, X, dropout=0.0):
        output = []
        backward = []
        start = 0
        for input_ in self.inputs:
            out, finish = input_.begin_update(X, dropout=drop)
            output.append(out)
            end = start + input_.nr_out
            backward.append((finish, start, end))
            start = end
        return self.ops.concatenate(output, axis=1), self._get_finish_update(backward)

    def _get_finish_update(self, backward):
        def finish_update(gradient):
            assert len(self.inputs) == 3 # TODO
            seq_grads = [bwd(gradient[:, start : end])
                         for bwd, start, end in backward]
            summed = []
            for grads in zip(*seq_grads):
                summed.append(sum(grads)) 
            return summed
        return finish_update
