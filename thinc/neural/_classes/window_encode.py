from collections import defaultdict

from .model import Model


class MaxoutWindowEncode(Model):
    nr_piece = 3
    nr_feat = 5
    nr_out = None
    nr_in = None

    @property
    def describe_params(self):
        init = self.ops.xavier_uniform_init
        yield 'W-%s' % self.name, self.shape, init
        yield 'b-%s' % self.name, (self.nr_out, self.nr_piece), None

    @property
    def shape(self):
        if self.output_shape is None or self.input_shape is None:
            return None
        else:
            return (self.nr_out, self.nr_piece, self.nr_feat, self.nr_in)

    @property
    def output_shape(self):
        return (self.nr_out,) if self.nr_out is not None else None

    @property
    def input_shape(self):
        return (self.nr_feat, self.nr_in) if self.nr_in is not None else None

    @property
    def W(self):
        return self.params.get('W-%s' % self.name, require=True)

    def __init__(self, nr_out, **kwargs):
        self.nr_out = nr_out
        Model.__init__(self, **kwargs)

    def initialize_params(self, train_data=None, add_gradient=True):
        if train_data is not None:
            self.nr_in = self._get_vector_dim_from_input(train_data)
        assert self.shape is not None, "TODO: Error"
        for name, shape, init in self.describe_params:
            if name not in self.params:
                self.params.add(name, shape)
                if init is not None:
                    init(self.params.get(name), inplace=True)

    def predict_batch(self, X):
        out, _ = self._forward(X)
        return out

    def begin_update(self, ids_vectors, dropout=0.0):
        ids, vectors = ids_vectors
        flat_out, whiches = self._forward(ids, vectors)
        flat_out, bp_dropout = self.ops.dropout(flat_out, dropout, inplace=True)
        finish_update = self._get_finish_update(ids, vectors, flat_out, whiches)
        return flat_out, bp_dropout(finish_update)

    def _forward(self, ids, vectors):
        positions, vectors = self._get_positions(ids, vectors)
        dotted = self._dot_ids(positions, vectors, [len(seq) for seq in ids])
        out = [self.ops.allocate((len(x), self.nr_out)) for x in ids]
        whiches = []
        for i, cands in enumerate(dotted):
            cands += self.b
            which = self.ops.argmax(cands)
            best = self.ops.take_which(cands, which)
            out[i][:] = best
            whiches.append(which)
        return out, whiches

    def _get_finish_update(self, vectors_BI, whiches_BO, lengths_B):
        B, I = vectors_BI.shape
        O = self.nr_out
        F = self.nr_feat
        P = self.nr_piece
        def finish_update(gradients_BO, optimizer=None, **kwargs):
            gradients_BOP = self.ops.allocate((B, O, P))
            _get_full_gradients(gradients_BOP, gradients_BO, whiches_BO)
            d_b = self.d_b
            d_b += gradients_BOP.sum(axis=0)
            
            inputs_BFI = self.ops.allocate((B, F, I))
            _get_full_inputs(inputs_BFI, vectors_BI, lengths_B)
            # Bop,Bfi->opfi
            d_W = self.d_W
            d_W += self.ops.batch_outer(gradients_BOP, inputs_BFI)
            return None
        return finish_update


def _dot_ids(ops, W, positions, vectors, lengths):
    # Shift the lengths, so that we don't have to special-case the starts and
    # ends. We'll shift afterwards.
    window_size = int((W.shape[1]-1) / 2)
    out = [ops.allocate((length+(window_size*2), W.shape[0], W.shape[-1]))
           for length in lengths]
    H__bopf = _compute_hidden_layer(ops, W, vectors, lengths)
    for id_, ijs in positions.items():
        for i, j in ijs:
            out[i][j : j+5] += H__bopf
    # Shift the output, to correct for the 'padding' shift above.
    return [arr[window_size : -window_size] for arr in out]


def _compute_hidden_layer(ops, W__opfi, vectors__bi, lengths):
    H__bopf = ops.xp.tensordot(vectors__bi, W__opfi, axes=[[1], [3]])
    H__bfop = H__bopf.transpose((0, 3, 1, 2))
    # Now zero features that were past boundaries, using the lengths
    _zero_features_past_sequence_boundaries(H__bfop, lengths)
    return H__bfop


def _get_positions(id_seqs, vector_seqs):
    assert len(id_seqs) == len(vector_seqs), \
        "TODO error %d vs %d" % (len(id_seqs), len(vector_seqs))
    positions = defaultdict(list)
    vectors_table = {}
    for i, (id_seq, vector_seq) in enumerate(zip(id_seqs, vector_seqs)):
        for j, (key, vector) in enumerate(zip(id_seq, vector_seq)):
            positions[key].append((i, j))
            vectors_table[key] = vector
    return positions, vectors_table


def _get_full_gradients(flat_gradients, gradients, whiches):
    for i in range(flat_gradients.shape[-1]):
        flat_gradients[:, :, i] += gradients * (whiches == i)
    return flat_gradients


def _get_full_inputs(inputs, vectors, lengths):
    # Col 0 has LL
    inputs[2:, 0] = vectors[:-2]
    # Col 1 has L
    inputs[1:, 1] = vectors[:-1]
    # Col 2 has w
    inputs[:, 2] = vectors
    # Col 3 has R
    inputs[:, 3] = vectors[1:]
    # Col 4 has RR
    inputs[:, 4] = vectors[2:]
    # Now use the lengths to zero LL, L, R and RR features as appropriate.
    i = 0
    for length in lengths:
        inputs[i + length-2, 3] = 0.
        inputs[i + length-1, 4] = 0.
        i += length
        if i >= 1:
            inputs[i-1, 1] = 0.
            if i >= 2:
                inputs[i-2, 2] = 0.
    return inputs


def _zero_features_past_sequence_boundaries(flat__bfop, lengths):
    # Now use the lengths to zero LL, L, R and RR features as appropriate.
    assert flat__bfop.shape[0] == sum(lengths), \
        (flat__bfop.shape, sum(lengths))
    i = 0
    for n in lengths:
        if n == 0:
            continue
        seq__nfop = flat__bfop[i : i+n]
        seq__nfop[0, 0] = 0
        seq__nfop[0, 1] = 0
        if len(seq__nfop) >= 2:
            seq__nfop[1, 0] = 0
        seq__nfop[-1, 3] = 0
        seq__nfop[-1, 4] = 0
        if len(seq__nfop) >= 2:
            seq__nfop[-2, 4] = 0
        i += n
