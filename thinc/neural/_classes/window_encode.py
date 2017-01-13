from collections import defaultdict

from .model import Model
from ... import describe
from ...describe import Dimension, Synapses, Biases, Gradient


def _set_dimensions_if_given(model, *args, **kwargs): # pragma: no cover 
    if len(args) >= 1:
        model.nO = args[0]
    elif not hasattr(model, 'nO'):
        model.nO = None
    if len(args) >= 2:
        model.nI = args[1]
    elif not hasattr(model, 'nI'):
        model.nI = None
    if 'pieces' in kwargs:
        model.nP = kwargs['pieces']
    if 'window' in kwargs:
        model.nF = kwargs['window']


def _set_dimensions_if_needed(model, X, y=None): # pragma: no cover
    if model.nI is None:
        model.nI = X.shape[0]
    if model.nO is None and y is not None:
        model.nO = y.max()


@describe.input(("nB", "nI"))
@describe.output(("nB", "nO"))
@describe.on_data(_set_dimensions_if_needed)
@describe.on_init(_set_dimensions_if_given)
@describe.attributes(
    nI=Dimension("Vector dimensionality"),
    nP=Dimension("Number of pieces"),
    nF=Dimension("Number of features"),
    nO=Dimension("Size of output"),
    W=Synapses("Weights matrix", ("nO", "nP", "nF", "nI"),
        lambda W, ops: ops.xavier_uniform_init(W)),
    b=Biases("Bias vector", ("nO", "nP")),
    d_W=Gradient("W"),
    d_b=Gradient("b")
)
class MaxoutWindowEncode(Model): # pragma: no cover
    def __init__(self, nr_out, embed, **kwargs):
        self.nr_out = nr_out
        self.embed = embed
        Model.__init__(self, **kwargs)

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
        cands = _dot_ids(self.ops, self.w.W, positions, vectors, lengths)
        cands += self.w.b
        which = self.ops.argmax(cands)
        best = self.ops.take_which(cands, which)
        return best, which

    def _get_finish_update(self, fine_tune, positions,
            vectors_UI, lengths, whiches_BO):
        B = whiches_BO.shape[0]
        I = self.nI
        O = self.nO
        F = self.nF
        P = self.nP
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
                W_OPFI = self.wW
                gradients_BFI = self.ops.xp.einsum(
                                    'bop,opfi->bfi', gradients_BOP, W_OPFI)
                gradients_BI = gradients_BFI.sum(axis=1)
                fine_tune(gradients_BI, optimizer=optimizer, **kwargs)
            return None
        return finish_update


def _get_uniq_vectors(positions, vectors): # pragma: no cover
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


def _get_positions(ids): # pragma: no cover
    positions = defaultdict(list)
    for i, id_ in enumerate(ids):
        positions[id_].append(i)
    return positions


def _dot_ids(ops, W, positions, vectors, lengths): # pragma: no cover
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


def _compute_hidden_layer(ops, W__opfi, vectors__ui, lengths): # pragma: no cover
    H__uopf = ops.xp.tensordot(vectors__ui, W__opfi, axes=[[1], [3]])
    H__ufop = H__uopf.transpose((0, 3, 1, 2))
    return H__ufop


def _get_full_gradients(flat_gradients, gradients, whiches): # pragma: no cover
    for i in range(flat_gradients.shape[-1]):
        flat_gradients[:, :, i] += gradients * (whiches == i)
    return flat_gradients


def _get_full_inputs(writeto, positions, vectors, lengths): # pragma: no cover
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


def _zero_features_past_sequence_boundaries(flat__bfop, lengths): # pragma: no cover
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
