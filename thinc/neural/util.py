from __future__ import print_function, unicode_literals
from .ops import NumpyOps, CupyOps
import time                                                


def get_ops(ops):
    if ops in ('numpy', 'cpu'):
        return NumpyOps()
    elif ops in ('cupy', 'gpu'):
        return CupyOps()
    else:
        raise ValueError("TODO error %s" % ops)


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
