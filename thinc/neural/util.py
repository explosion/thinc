from __future__ import print_function, unicode_literals
import numpy
from preshed.maps import PreshMap
from .ops import NumpyOps, CupyOps


def get_ops(ops):
    if ops in ('numpy', 'cpu'):
        return NumpyOps()
    elif ops in ('cupy', 'gpu'):
        return CupyOps()
    else:
        raise ValueError("TODO error %s" % ops)


def mark_sentence_boundaries(sequences, drop=0.): # pragma: no cover
    '''Pad sentence sequences with EOL markers.'''
    for sequence in sequences:
        sequence.insert(0, '-EOL-')
        sequence.insert(0, '-EOL-')
        sequence.append('-EOL-')
        sequence.append('-EOL-')
    return sequences, None


def remap_ids(ops):
    id_map = PreshMap()
    def begin_update(ids, drop=0.):
        return ops.remap_ids(id_map, ids, len(id_map)), None
    return begin_update

#    def _unique_ids(self, ids):
#        id_map = {}
#        for i, id_ in enumerate(ids.flatten()):
#            if id_ not in id_map:
#                id_map[id_] = [i]
#            else:
#                id_map[id_].append(i)
#        # Currently this is handled on CPU anyway, so allocate on CPU.
#        uniques = numpy.asarray(sorted(id_map.keys()), dtype='uint64')
#        return uniques, id_map


def flatten_sequences(sequences, drop=0.): # pragma: no cover
    ops = NumpyOps()
    return ops.flatten(sequences), None


def partition(examples, split_size): # pragma: no cover
    examples = list(examples)
    numpy.random.shuffle(examples)
    n_docs = len(examples)
    split = int(n_docs * split_size)
    return examples[:split], examples[split:]


def minibatch(stream, batch_size=1000): # pragma: no cover
    batch = []
    for X in stream:
        batch.append(X)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) != 0:
        yield batch
