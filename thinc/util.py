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
    y = model.ops.flatten(y)
    scores = model.predict_batch(X)
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
