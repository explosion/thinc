from collections import defaultdict
import numpy


def linear_decay(rate, decay, nr_upd):
    return rate * 1./(1. + decay * nr_upd)


def update_averages(averages, key, weights, nr_upd, max_decay=0.9999):
    decay = (1. + nr_upd) / (10. + nr_upd)
    decay = min(decay, max_decay)

    if key not in averages:
        averages[key] = numpy.zeros(weights.shape)
    avg = averages[key]
    avg -= (1-decay) * (avg - weights)


def clip_gradient(gradient, threshold):
    grad_norm = numpy.linalg.norm(gradient)
    if grad_norm >= threshold:
        gradient *= threshold / grad_norm


class SGD(object):
    def __init__(self, ops, lr, momentum=0.0, decay=0.0, **settings):
        self.ops = ops
        self.alpha = lr
        self.mu = momentum
        self.decay = decay
        self.momentums = {}
        self.averages = {}
        self.nr_update = defaultdict(int)

    def __call__(self, weights, gradient, key=None):
        self.nr_update[key] += 1
        nr_upd = self.nr_update[key]
        lr = self.lr(nr_upd)
        if key is None or self.mu == 0.0:
            weights -= lr * gradient
            gradient.fill(0)
        else:
            if key not in self.momentums:
                self.momentums[key] = numpy.zeros(weights.shape)
            momentum = self.momentums[key]
            momentum *= self.mu
            momentum += gradient * lr
            weights -= momentum
            gradient.fill(0)
        update_averages(self.averages, key, weights, nr_upd)

    def lr(self, nr_upd):
        return linear_decay(self.alpha, self.decay, nr_upd)


class Adam(object):
    def __init__(self, ops, lr, beta1=0.90, beta2=0.999, eps=1e-08, decay=0.0):
        self.ops = ops
        self.mom1 = {}
        self.mom2 = {}
        self.averages = {}
        self.nr_update = defaultdict(int)
        self.last_seen = defaultdict(int)
        self.alpha = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.decay = decay
        self.d = 1.
        self.f = 0.

    def lr(self, nr_upd):
        alpha = linear_decay(self.alpha, self.decay, nr_upd)
        fix1 = 1.- (self.b1 ** nr_upd)
        fix2 = 1.- (self.b2 ** nr_upd)
        return alpha * numpy.sqrt(fix2) / fix1

    @property
    def nr_iter(self):
        if not self.nr_update:
            return 0
        return max(self.nr_update.values())

    def __call__(self, weights, gradient, key=None):
        assert key is not None
        assert len(gradient) >= 1
        assert not self.ops.xp.isnan(weights).any()
        assert not self.ops.xp.isnan(gradient).any()
        if key not in self.mom1:
            self.mom1[key] = self.ops.allocate(weights.shape)
        if key not in self.mom2:
            self.mom2[key] = self.ops.allocate(weights.shape)
        self.nr_update[key] += 1
        nr_upd = self.nr_update[key]

        clip_gradient(gradient, len(gradient) / 100.)
        mom1 = self.mom1[key]
        mom1 *= self.b1
        mom1 += (gradient * (1-self.b1))

        mom2 = self.mom2[key]
        mom2 *= self.b2
        mom2 += (1-self.b2) * gradient ** 2
        
        lr = self.lr(nr_upd)
        weights -= lr * mom1 / (self.d * numpy.sqrt(mom2) + self.eps)
        gradient.fill(0)
        update_averages(self.averages, key, weights, nr_upd)

    def set_loss(self, loss):
        pass


class Eve(Adam):
    def __init__(self, *args, **kwargs):
        Adam.__init__(self, *args, **kwargs)
        self.b3 = 0.999
        self.lower_threshold = 0.1
        self.upper_threshold = 10
        self.d = 1.
        self.f = None

    def set_loss(self, loss):
        if self.f is None:
            self.f = loss
            return
        old_f = self.f
        d = self.d
        c = self._get_c(loss, old_f)
        new_f = c * loss
        r = abs(new_f - old_f) / min(new_f, old_f)
        new_d = d + (1 - self.b3) * (r - d)
        self.d = new_d
        self.f = new_f

    def _get_c(self, loss, old_f):
        if loss < old_f:
            delta = self.lower_threshold + 1.
            Delta = self.upper_threshold + 1.
        else:
            delta = 1. / (self.upper_threshold + 1.)
            Delta = 1. / (self.lower_threshold + 1.)
        return min(max(delta, loss / old_f), Delta)
