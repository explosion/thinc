from __future__ import print_function, unicode_literals
from .ops import NumpyOps, CupyOps
import time                                                

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        model, X, y = args
        n_y = len(y)
        print('%r %d examples %2.2f sec' % (model.name, n_y, te-ts))
        return result

    return timed


def get_ops(ops):
    if ops in ('numpy', 'cpu'):
        return NumpyOps()
    elif ops == ('cupy', 'gpu'):
        return CupyOps()
    else:
        return ops


@timeit
def score_model(model, X, y):
    correct = 0
    total = 0
    scores = model.predict_batch(X)
    if isinstance(y, tuple) and (isinstance(y[0], tuple) or isinstance(y[0], list)):
        y = model.ops.asarray(model.ops.flatten(y), dtype='i')
    else:
        y = model.ops.asarray(y, dtype='i')
    for i, gold in enumerate(y):
        correct += scores[i].argmax() == gold
        total += 1
    return float(correct) / total


def partition(examples, split_size):
    examples = list(examples)
    numpy.random.shuffle(examples)
    n_docs = len(examples)
    split = int(n_docs * split_size)
    return examples[:split], examples[split:]


def minibatch(stream, batch_size=1000):
    batch = []
    for X in stream:
        batch.append(X)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) != 0:
        yield batch


def im2col(sequence, window):
    assert window == 2
    output = numpy.zeros((sequence.shape[0], window*2+1, sequence.shape[1]))
    output[:, 2] = sequence
    output[2:, 0] = sequence[:-2]
    output[1:, 1] = sequence[:-1]
    output[:-1, 3] = sequence[1:]
    output[:-2, 4] = sequence[2:]
    # Words 0 and 1 have no LL feature
    assert sum(abs(output[0, 0])) == 0
    if len(output) >= 2:
        assert sum(abs(output[1, 0])) == 0
        # Word 0 no L feature
        assert sum(abs(output[1, 0])) == 0
    # Words -1 and -2 have no RR feature
    assert sum(abs(output[-1, 4])) == 0
    if len(output) >= 2:
        assert sum(abs(output[-2, 4])) == 0
        # Word -1 has no R feature
    assert sum(abs(output[-1, 3])) == 0
    return output


class Unassigned(object):
    def __init__(self, expected_type):
        self.expected_type = expected_type

    def __get__(self, obj, objtype):
        return None

    def __set__(self, obj, value):
        if not isinstance(obj, self.expected_type):
            raise TypeError("TODO Error")
