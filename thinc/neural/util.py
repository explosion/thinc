from __future__ import print_function, unicode_literals
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


def flatten_sequences(sequences, drop=0.):
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
