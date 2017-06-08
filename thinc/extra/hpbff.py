import numpy.random
import copy


def minibatch(train_X, train_y, size=16):
    indices = numpy.arange(len(train_X))
    numpy.random.shuffle(indices)
    j = 0
    while j < indices.shape[0]:
        slice_ = indices[j : j + size]
        X = _take_slice(train_X, slice_)
        y = _take_slice(train_y, slice_)
        yield X, y
        j += size

def _take_slice(data, slice_):
    if isinstance(data, list) or isinstance(data, tuple):
        return [data[int(i)] for i in slice_]
    else:
        return data[slice_]


def best_first_sgd(initials, train_X, train_y, dev_X, dev_y,
        get_new_model=None, get_score=None):
    if get_new_model is None:
        get_new_model = _get_new_model
    if get_score is None:
        get_score = _get_score

    queue = []
    for i, model in enumerate(initials):
        train_acc, model = get_new_model(model, train_X, train_y)
        check_acc = get_score(model, dev_X, dev_y)
        ratio = min(check_acc / train_acc, 1.0)
        print((model[-1], train_acc, check_acc))
        queue.append([check_acc * ratio, i, model])
 
    train_acc = 0
    limit = 8
    i = 0
    best_model = None
    best_acc = 0.0
    best_i = 0
    while best_i > (i - 100) and train_acc < 0.999:
        queue.sort(reverse=True)
        queue = queue[:limit]
        prev_score, parent, model = queue[0]
        queue[0][0] -= 0.0005
        train_acc, new_model = get_new_model(model, train_X, train_y)
        check_acc = get_score(new_model, dev_X, dev_y)
        ratio = min(check_acc / train_acc, 1.0)
            
        i += 1
        queue.append([check_acc * ratio, i, new_model])
   
        if check_acc >= best_acc:
            best_acc = check_acc
            best_i = i
            best_model = new_model
        progress = {
            'i': i,
            'parent': parent,
            'prev_score': prev_score,
            'this_score': queue[-1][0],
            'train_acc': train_acc,
            'check_acc': check_acc,
            'best_acc': best_acc,
            'hparams': new_model[-1]
        }
        yield best_model, progress


def _get_new_model(model_sgd_hparams, train_X, train_y):
    model, sgd, hparams = copy.deepcopy(model_sgd_hparams)

    hparams['learn_rate'] = resample(hparams['learn_rate'], 1e-6, 0.1)
    hparams['L2'] = resample(hparams['L2'], 0.0, 1e-3)
    hparams['batch_size'] = int(resample(hparams['batch_size'], 1, 256))
    hparams['epochs'] = hparams.get('epochs', 0) + 1
    if hparams['epochs'] >= 5:
        hparams['dropout'] = resample(hparams['dropout'], 0.05, 0.7)

    train_acc = 0.
    for X, y in minibatch(train_X, train_y, size=hparams['batch_size']):
        yh, finish_update = model.begin_update(X, drop=hparams['dropout'])
        dy = (yh-y) / y.shape[0]
        finish_update(dy, sgd=sgd)
        train_acc += (y.argmax(axis=1) == yh.argmax(axis=1)).sum()
    train_acc /= train_y.shape[0]
    return train_acc, (model, sgd, hparams)


def _get_score(model, dev_X, dev_y):
    with model[0].use_params(model[1].averages):
        score = model[0].evaluate(dev_X, dev_y)
    return score


def resample(curr, min_, max_):
    scale = (max_ - min_) / 10
    next_ = numpy.random.normal(loc=curr, scale=scale)
    return min(max_, max(min_, next_))
