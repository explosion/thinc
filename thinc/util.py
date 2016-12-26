def get_ops(ops):
    if ops == 'numpy':
        return NumpyOps()
    elif ops == 'cupy':
        return CupyOps()
    else:
        return ops


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
