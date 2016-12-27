from .base import Model


class WindowEncode(Model):
    def __init__(self, vectors=None, W=None, b=None, ops=None):
        self.ops = ops
        self.vectors = vectors
        self.W = W

    def predict_batch(self, batch):
        out, _ = self._forward(X)
        return out

    def begin_update(self, ids, dropout=0.0):
        batch_outputs, whiches = self._forward(X)
        mask = dropout(batch_outputs, drop, inplace=True)
        finish_update = self._get_finish_output(ids, batch_outputs, whiches, mask)
        return batch_outputs

    def _forward(self, batch):
        positions = self._get_positions(batch)
        dotted = self._dot_ids(positions, [len(seq) for seq in batch])
        out = [self.ops.allocate((len(x), self.nr_out)) for x in batch]
        for i, cands in enumerate(dotted):
            cands += self.b
            which = self.ops.argmax(cands)
            best = self.ops.take_which(cands, which)
            out[i][:] = best
        return out, whiches

    def _get_positions(self, batch):
        ids = defaultdict(list)
        for i, seq in enumerate(batch):
            for j, id_ in enumerate(seq):
                ids[id_].append((i, j))
        for id_ in ids:
            vector = self.get_param(id_)
            if vector is None:
                self.add_param(id_, (self.nr_in,))
        return ids

    def _get_finish_update(self, batch_outputs, whiches):
        def finish_update(batch_gradients):
            if drop:
                for grad, mask in zip(batch_gradients, drop_mask):
                    grad *= mask
            all_inputs = self._get_all_inputs(X)
            all_gradients = self._get_all_gradients(batch_outputs, batch_gradients,
                                                    whiches)
            if all_inputs.shape[0] == 0 or all_gradients.shape[0] == 0:
                return None
            self.d_b += all_gradients.sum(axis=0)
            # Bop,Bfi->opfi
            self.d_W += self.ops.batch_outer(all_gradients, all_inputs)
            tuned_ids = self._fine_tune(self.W, X, all_gradients)
            for id_ in tuned_ids:
                optimizer(self.get_param(id_), self.get_gradient(id_), key=id_)
            return None
        return finish_update

    def _dot_ids(self, ids, lengths):
        out = [self.allocate((length, self.nr_out, self.nr_piece)) for length in lengths]
        for id_, egs in ids.items():
            vector = self.get_param(id_)
            if vector is None:
                continue
            # opFi,i->Fop
            hidden = numpy.tensordot(self.W, vector, axes=[[3], [0]])
            hidden = hidden.transpose((2, 0, 1))
            for i, j in egs:
                out_i = out[i]
                if j >= 2 and (j+2) < lengths[i]:
                    out_i[j-2:j+3] += hidden
                else:
                    if j >= 2:
                        out_i[j-2] += hidden[0]
                    if j >= 1:
                        out_i[j-1] += hidden[1]
                    out_i[j] += hidden[2]
                    if (j+1) < lengths[i]:
                        out_i[j+1] += hidden[3]
                    if (j+2) < lengths[i]:
                        out_i[j+2] += hidden[4]
        return out

    def _fine_tune(self, W, X, all_gradients):
        # opfi,Bop->Bfi
        tuning = numpy.tensordot(all_gradients, W, axes=[[1,2], [0, 1]])
        tuned_ids = set()
        i = 0
        for ids in X:
            for id_ in ids:
                d_vector = self.get_gradient(id_)
                if d_vector is None:
                    continue
                if i >= 2:
                    d_vector += tuning[i-2, 0]
                if i >= 1:
                    d_vector += tuning[i-1, 1]
                d_vector += tuning[i, 2]
                if (i+1) < tuning.shape[0]:
                    d_vector += tuning[i+1, 3]
                if (i+2) < tuning.shape[0]:
                    d_vector += tuning[i+2, 4]
                i += 1
                tuned_ids.add(id_)
        return tuned_ids

    def _get_all_inputs(self, X):
        total_length = sum(len(x) for x in X)
        all_inputs = self.ops.allocate((total_length, 5, self.nr_in))
        i = 0
        for ids in X:
            vectors = [self.get_param(id_) for id_ in ids]
            for vector in vectors:
                if i >= 2:
                    all_inputs[i-2, 0] = vector
                if i >= 1:
                    all_inputs[i-1, 1] = vector
                all_inputs[i, 2] = vector
                if (i+1) < total_length:
                    all_inputs[i+1, 3] = vector
                if (i+2) < total_length:
                    all_inputs[i+2, 4] = vector
                i += 1
        return all_inputs

    def _get_all_gradients(self, batch_outputs, batch_gradients, whiches):
        assert len(batch_outputs) == len(batch_gradients)
        total_length = sum(len(x) for x in batch_outputs)
        all_gradients = self.allocate((total_length, self.nr_out, self.nr_piece))
        i = 0
        for output, gradients, which in zip(batch_outputs, batch_gradients, whiches):
            assert output.shape == gradients.shape, (output.shape, gradients.shape)
            for grad, wh in zip(gradients, which):
                for j in range(self.nr_piece):
                    all_gradients[i, :, j] += grad * (wh == j)
                i += 1
        return all_gradients
