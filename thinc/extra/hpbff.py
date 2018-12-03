# coding: utf8
from __future__ import unicode_literals

import numpy
import numpy.random
import itertools
import srsly
import tqdm


def minibatch(train_X, train_y, size=16, nr_update=1000):
    with tqdm.tqdm(total=nr_update * size, leave=False) as pbar:
        while nr_update >= 0:
            indices = numpy.arange(len(train_X))
            numpy.random.shuffle(indices)
            j = 0
            while j < indices.shape[0]:
                slice_ = indices[j : j + size]
                X = _take_slice(train_X, slice_)
                y = _take_slice(train_y, slice_)
                yield X, y
                j += size
                nr_update -= 1
                if nr_update <= 0:
                    break
                pbar.update(size)


def _take_slice(data, slice_):
    if isinstance(data, list) or isinstance(data, tuple):
        return [data[int(i)] for i in slice_]
    else:
        return data[slice_]


class BestFirstFinder(object):
    def __init__(self, **param_values):
        self.queue = []
        self.limit = 16
        self.params = param_values
        self.best_acc = 0.0
        self.best_i = 0
        self.i = 0
        self.j = 0
        self.best_model = None
        self.temperature = 0.0

    @property
    def configs(self):
        keys, value_groups = zip(*self.params.items())
        for values in itertools.product(*value_groups):
            config = dict(zip(keys, values))
            yield config

    def enqueue(self, model, train_acc, check_acc):
        fom = check_acc * min(check_acc / train_acc, 1.0)
        self.queue.append([fom, self.i, 0, model])
        if check_acc >= self.best_acc:
            self.best_acc = check_acc
            self.best_i = self.i
            self.best_model = model
            self.temperature = 0.0
        else:
            self.temperature += 0.01
        self.j = 0
        self.queue.sort(reverse=True)
        self.queue = self.queue[: self.limit]

    def __iter__(self):
        self.queue.sort(reverse=True)
        self.queue = self.queue[: self.limit]
        for i in range(len(self.queue)):
            self.queue[i][0] = self.queue[i][0] - 0.01
            self.queue[i][-1][2]["parent"] = self.queue[i][2]
            self.queue[i][2] += 1
            yield self.queue[i][-1]

    @property
    def best(self):
        return self.best_model


def resample_hyper_params(hparams, temperature):
    hparams = dict(hparams)
    hparams["epochs"] = hparams.get("epochs", 0) + 1
    hparams["learn_rate"] = resample(hparams["learn_rate"], 1e-6, 0.1, temperature)
    # hparams['beta1'] = resample(hparams.get('beta1', 0.9), 0.8, 1.0, temperature)
    # hparams['beta2'] = resample(hparams.get('beta2', 0.9), 0.8, 1.0, temperature)
    # hparams['L2'] = resample(hparams['L2'], 0.0, 1e-3, temperature)
    # hparams['batch_size'] = int(resample(hparams['batch_size'], 10, 256, temperature))
    # hparams['dropout'] = resample(hparams['dropout'], 0.05, 0.7, temperature)
    return hparams


def resample(curr, min_, max_, temperature):
    if temperature == 0.0:
        return curr
    scale = (max_ - min_) * temperature
    next_ = numpy.random.normal(loc=curr, scale=scale)
    return min(max_, max(min_, next_))


def train_epoch(
    model, sgd, hparams, train_X, train_y, dev_X, dev_y, device_id=-1, temperature=0.0
):
    model, sgd, hparams = srsly.pickle_loads(srsly.pickle_dumps((model, sgd, hparams)))
    if device_id >= 0:
        model.to_gpu(device_id)
        sgd.ops = model.ops
        sgd.to_gpu()
        if isinstance(train_y, numpy.ndarray):
            train_y = model.ops.asarray(train_y)
            dev_y = model.ops.asarray(dev_y)
    hparams = resample_hyper_params(hparams, temperature)
    sgd.learn_rate = hparams["learn_rate"]
    sgd.beta1 = hparams["beta1"]
    sgd.beta2 = hparams["beta2"]
    sgd.L2 = hparams["L2"]
    train_acc = 0.0
    train_n = 0
    for X, y in minibatch(
        train_X, train_y, size=hparams["batch_size"], nr_update=hparams["nr_update"]
    ):
        yh, finish_update = model.begin_update(X, drop=hparams["dropout"])
        if hasattr(y, "shape"):
            dy = (yh - y) / y.shape[0]
            train_acc += (y.argmax(axis=1) == yh.argmax(axis=1)).sum()
            train_n += y.shape[0]
        else:
            n_y = sum(len(y_i) for y_i in y)
            dy = [(yh[i] - y[i]) / n_y for i in range(len(yh))]
            for i in range(len(y)):
                train_acc += (y[i].argmax(axis=1) == yh[i].argmax(axis=1)).sum()
            train_n += n_y
        finish_update(dy, sgd=sgd)
    train_acc /= train_n
    with model.use_params(sgd.averages):
        dev_acc = model.evaluate(dev_X, dev_y)
    model.to_cpu()
    sgd.to_cpu()
    return device_id, ((model, sgd, hparams), float(train_acc), float(dev_acc))


class DevicePool(object):
    """Synchronize GPU usage"""

    def __init__(self, n):
        self.devices = {i: None for i in range(n)}

    def acquire(self):
        for i, device in self.devices.items():
            if device is None:
                self.devices[i] = True
                return i
        else:
            return None

    def release(self, i):
        if i in self.devices:
            self.devices[i] = None


#
# def best_first_sgd(initials, train_X, train_y, dev_X, dev_y,
#        get_new_model=None, get_score=None):
#    if get_new_model is None:
#        get_new_model = _get_new_model
#    if get_score is None:
#        get_score = _get_score
#
#    queue = []
#    for i, model in enumerate(initials):
#        train_acc, model = get_new_model(model, train_X, train_y)
#        check_acc = get_score(model, dev_X, dev_y)
#        ratio = min(check_acc / train_acc, 1.0)
#        print((model[-1], train_acc, check_acc))
#        queue.append([check_acc * ratio, i, model])
#
#    train_acc = 0
#    limit = 8
#    i = 0
#    best_model = None
#    best_acc = 0.0
#    best_i = 0
#    while best_i > (i - 100) and train_acc < 0.999:
#        queue.sort(reverse=True)
#        queue = queue[:limit]
#        prev_score, parent, model = queue[0]
#        queue[0][0] -= 0.001
#        yield prev_score, parent, model
#        train_acc, new_model = get_new_model(model, train_X, train_y)
#        check_acc = get_score(new_model, dev_X, dev_y)
#        ratio = min(check_acc / train_acc, 1.0)
#
#        i += 1
#        queue.append([check_acc * ratio, i, new_model])
#
#        if check_acc >= best_acc:
#            best_acc = check_acc
#            best_i = i
#            best_model = new_model
#        progress = {
#            'i': i,
#            'parent': parent,
#            'prev_score': prev_score,
#            'this_score': queue[-1][0],
#            'train_acc': train_acc,
#            'check_acc': check_acc,
#            'best_acc': best_acc,
#            'hparams': new_model[-1]
#        }
#        yield best_model, progress
#
#
#
