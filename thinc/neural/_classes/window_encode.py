from collections import defaultdict

from .model import Model


class MaxoutWindowEncode(Model):
    name = 'encode'
    nr_piece = 3
    nr_feat = 5
    nr_out = None
    nr_in = None

    @property
    def describe_params(self):
        def _init(W, inplace=True):
            fan_in = W.shape[2] * W.shape[3]
            fan_out = W.shape[0]
            for i in range(W.shape[1]):
                scale = self.ops.xp.sqrt(2. / (fan_in + fan_out))
                W[:, i] = self.ops.xp.random.uniform(-scale, scale,
                                                     W[:, i].shape)
            return W
        yield 'W-%s' % self.name, self.shape, _init
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
    
    @property
    def b(self):
        return self.params.get('b-%s' % self.name, require=True)

    @property
    def d_W(self):
        return self.params.get('d_W-%s' % self.name, require=True)
    
    @property
    def d_b(self):
        return self.params.get('d_b-%s' % self.name, require=True)

    def __init__(self, nr_out, embed, **kwargs):
        self.nr_out = nr_out
        self.embed = embed
        Model.__init__(self, **kwargs)

    def initialize_params(self, train_data=None, add_gradient=True):
        if train_data is not None and self.nr_in is None:
            self.nr_in = self._get_vector_dim_from_input(train_data)
        assert self.shape is not None, "TODO: Error"
        for name, shape, init in self.describe_params:
            if name not in self.params:
                self.params.add(name, shape)
                if init is not None:
                    init(self.params.get(name), inplace=True)
        self.embed.initialize_params(train_data, add_gradient=add_gradient)

    def predict_batch(self, seqs):
        positions, vectors, lengths = self.embed.predict_batch(seqs)
        out, _ = self._forward(positions, vectors, lengths)
        return out

    def begin_update(self, seqs, dropout=0.0):
        (positions, vectors, lengths), fine_tune = self.embed.begin_update(seqs, dropout=dropout)
        flat_out, whiches = self._forward(positions, vectors, lengths)
        flat_out, bp_dropout = self.ops.dropout(flat_out, dropout, inplace=True)
        finish_update = self._get_finish_update(
                            fine_tune, positions, vectors, lengths, whiches)
        return flat_out, bp_dropout(finish_update)

    def _forward(self, positions, vectors, lengths):
        cands = _dot_ids(self.ops, self.W, positions, vectors, lengths)
        cands += self.b
        which = self.ops.argmax(cands)
        best = self.ops.take_which(cands, which)
        return best, which

    def _get_finish_update(self, fine_tune, positions, vectors_UI, lengths, whiches_BO):
        B = whiches_BO.shape[0]
        I = vectors_UI.shape[1]
        O = self.nr_out
        F = self.nr_feat
        P = self.nr_piece
        def finish_update(gradients_BO, optimizer=None, **kwargs):
            gradients_BOP = self.ops.allocate((B, O, P))
            _get_full_gradients(gradients_BOP, gradients_BO, whiches_BO)
            d_b = self.d_b
            d_b += gradients_BOP.sum(axis=0)
            
            inputs_BFI = self.ops.allocate((B, F, I))
            _get_full_inputs(inputs_BFI, positions, vectors_UI, lengths)
            # Bop,Bfi->opfi
            d_W = self.d_W
            d_W += self.ops.batch_outer(gradients_BOP, inputs_BFI)
            if fine_tune is not None:
                W_OPFI = self.W
                gradients_BFI = self.ops.xp.einsum(
                                    'bop,opfi->bfi', gradients_BOP, W_OPFI)
                gradients_BI = gradients_BFI.sum(axis=1)
                fine_tune(gradients_BI, optimizer=optimizer, **kwargs)
            return None
        return finish_update


def _get_uniq_vectors(positions, vectors):
    # Need to set id back into sequence
    # Or, really, need to fix API of WindowEncode tagger. Maybe insert
    # shim operation?
    uniq_vectors = []
    remapped_positions = {}
    for i, (key, key_positions) in enumerate(positions.items()):
        seq_idx = key_positions[0]
        uniq_vectors.append(vectors[seq_idx])
        remapped_positions[i] = key_positions
    return remapped_positions, uniq_vectors


def _get_positions(ids):
    positions = defaultdict(list)
    for i, id_ in enumerate(ids):
        positions[id_].append(i)
    return positions


def _dot_ids(ops, W, positions, vectors, lengths):
    window_size = int((W.shape[2]-1) / 2)
    total_length = sum(lengths)
    # Shift the input, so that we don't have to special-case the starts and
    # ends. We'll shift back afterwards.
    out__bop = ops.allocate((total_length+(window_size*2), W.shape[0], W.shape[1]))
    H__ufop = _compute_hidden_layer(ops, W, vectors, lengths)
    
    for vec_idx, tok_idxs in positions.items():
        v__fop = H__ufop[vec_idx]
        for i in tok_idxs:
            out__bop[i : i+5] += v__fop
    # Shift the output, to correct for the 'padding' shift above.
    out__bop = out__bop[window_size : -window_size]
    return out__bop


def _compute_hidden_layer(ops, W__opfi, vectors__ui, lengths):
    H__uopf = ops.xp.tensordot(vectors__ui, W__opfi, axes=[[1], [3]])
    H__ufop = H__uopf.transpose((0, 3, 1, 2))
    return H__ufop


def _get_full_gradients(flat_gradients, gradients, whiches):
    for i in range(flat_gradients.shape[-1]):
        flat_gradients[:, :, i] += gradients * (whiches == i)
    return flat_gradients


def _get_full_inputs(writeto, positions, vectors, lengths):
    for vec_idx, tok_idxs in sorted(positions.items()):
        writeto[vec_idx, 2] = vectors[vec_idx]
    # Col 2 has w
    vectors = writeto[:, 2]
    # Col 0 has LL
    writeto[2:, 0] = vectors[:-2]
    # Col 1 has L
    writeto[1:, 1] = vectors[:-1]
    # Col 3 has R
    writeto[:-1, 3] = vectors[1:]
    # Col 4 has RR
    writeto[:-2, 4] = vectors[2:]
    ## Now use the lengths to zero LL, L, R and RR features as appropriate.
    #_zero_features_past_sequence_boundaries(writeto, lengths)
    return writeto


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
