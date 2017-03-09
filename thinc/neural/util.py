from __future__ import print_function, unicode_literals
import numpy
from preshed.maps import PreshMap
from .ops import NumpyOps, CupyOps

try:
    import cupy
    from cupy import get_array_module
except ImportError:
    cupy = None
    get_array_module = lambda _: numpy


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
    id_map = {0: 0}
    def begin_update(ids, drop=0.):
        n_vector = len(id_map)
        for i, id_ in enumerate(ids):
            if id_ not in id_map:
                id_map[id_] = n_vector
                n_vector += 1
            ids[i] = id_map[id_]
        return ids, None
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


def to_categorical(y, nb_classes=None):
    # From keras
    xp = get_array_module(y)
    if xp is cupy:
        y = y.get()
    y = numpy.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = numpy.max(y) + 1
    n = y.shape[0]
    categorical = numpy.zeros((n, nb_classes), dtype='float32')
    categorical[numpy.arange(n), y] = 1
    return xp.asarray(categorical)


def flatten_sequences(sequences, drop=0.): # pragma: no cover
    xp = get_array_module(sequences[0])
    return xp.concatenate(sequences), None


def partition(examples, split_size): # pragma: no cover
    examples = list(examples)
    numpy.random.shuffle(examples)
    n_docs = len(examples)
    split = int(n_docs * split_size)
    return examples[:split], examples[split:]


def minibatch(stream, batch_size=1000): # pragma: no cover
    if hasattr(stream, '__len__') and hasattr(stream, '__getitem__'):
        i = 0
        while i < len(stream):
            yield stream[i : i + batch_size]
            i += batch_size
    else:
        batch = []
        for X in stream:
            batch.append(X)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if len(batch) != 0:
            yield batch
