from ._classes.model import Model


class MeanPooling(Model):
    name = 'mean-pool'
    def predict(self, seqs):
        X = self.ops.xp.vstack(seqs)
        lengths = [len(seq) for seq in seqs]
        means = self.ops.allocate((len(lengths), X.shape[1]))
        start = 0
        for i, length in enumerate(lengths):
            end = start + length
            means[i] = X[start : end].mean(axis=0)
            start = end
        assert means.shape == (len(seqs), seqs[0].shape[1])
        return means

    def begin_update(self, seqs, drop=0.0):
        X = self.ops.xp.vstack(seqs)
        lengths = [len(seq) for seq in seqs]
        #X, bp_dropout = self.ops.dropout(X, drop)
        def finish_update(gradient, sgd=None):
            batch_grads = self.ops.allocate(X.shape)
            start = 0
            for i, length in enumerate(lengths):
                end = start + length
                batch_grads[start : end] += gradient[i] / (end-start)
                start = end
            return self.ops.unflatten(batch_grads, lengths)
        return self.predict(seqs), finish_update #bp_dropout(finish_update)


class MaxPooling(Model):
    name = 'max-pool'
    def predict(self, X):
        maxes = []
        for x in X:
            if x.shape[0] == 0:
                maxes.append(self.ops.allocate(x.shape[1:]))
            else:
                maxes.append(x.max(axis=0))
        return self.ops.asarray(maxes)

    def begin_update(self, X, drop=0.0):
        X, bp_dropout = self.ops.dropout(X, drop)
        def finish_update(gradient, sgd=None):
            batch_grads = []
            for i, x in enumerate(X):
                grad_i = self.ops.allocate(x.shape)
                if x.shape[0] != 0:
                    grad_i += gradient[i] * (x == x.max(axis=0))
                batch_grads.append(grad_i)
            return batch_grads
        return self.predict(X), bp_dropout(finish_update)


class MinPooling(Model):
    name = 'min-pool'
    def predict(self, X):
        maxes = []
        for x in X:
            if x.shape[0] == 0:
                maxes.append(self.ops.allocate(x.shape[1:]))
            else:
                maxes.append(x.min(axis=0))
        return self.ops.asarray(maxes)

    def begin_update(self, X, drop=0.0):
        X, bp_dropout = self.ops.dropout(X, drop)
        def finish_update(gradient, sgd=None):
            batch_grads = []
            for i, x in enumerate(X):
                grad_i = self.ops.allocate(x.shape)
                if x.shape[0] != 0:
                    grad_i += gradient[i] * (x == x.min(axis=0))
                batch_grads.append(grad_i)
            return batch_grads
        return self.predict(X), bp_dropout(finish_update)


class MultiPooling(Model): # pragma: no cover
    name = 'multi-pool'
    def __init__(self, *inputs):
        self.inputs = inputs

    def predict(self, X):
        return self.ops.xp.hstack([in_.predict(X) for in_ in self.inputs])

    def begin_update(self, X, drop=0.0):
        output = []
        backward = []
        start = 0
        length = X[0].shape[1]
        for input_ in self.inputs:
            out, finish = input_.begin_update(X, drop=drop)
            output.append(out)
            end = start + length
            backward.append((finish, start, end))
            start = end
        return self.ops.xp.hstack(output), self._get_finish_update(backward)

    def _get_finish_update(self, backward):
        def finish_update(gradient, sgd=None):
            assert len(self.inputs) == 3 # TODO
            seq_grads = [bwd(gradient[:, start : end], sgd=None)
                         for bwd, start, end in backward]
            summed = []
            for grads in zip(*seq_grads):
                summed.append(sum(grads))
            return summed
        return finish_update
