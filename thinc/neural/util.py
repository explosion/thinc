from .ops import NumpyOps, CupyOps


def get_ops(ops):
    if ops in ('numpy', 'cpu'):
        return NumpyOps()
    elif ops == ('cupy', 'gpu'):
        return CupyOps()
    else:
        return ops


def score_model(model, X_y):
    correct = 0
    total = 0
    X, y = zip(*X_y)
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
