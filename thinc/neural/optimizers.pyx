# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libc.stdlib cimport calloc, malloc, free

from collections import defaultdict
import numpy

from ..typedefs cimport weight_t
from .ops import NumpyOps, CupyOps, add_gradient_noise
from .util import get_array_module


def linear_decay(rate, decay, nr_upd):
    return rate * 1./(1. + decay * nr_upd)


class SGD(object):
    def __init__(self, ops, lr, momentum=0.0, nesterov=False, decay=0.0, **settings):
        self.ops = ops
        self.alpha = lr
        self.mu = momentum
        self.decay = decay
        self.nesterov = nesterov
        self.max_grad_norm = 100.
        self.momentums = {}
        self.averages = {} if settings.get('averages', True) else None
        self.nr_update = defaultdict(int)

    @property
    def nr_iter(self):
        if not self.nr_update:
            return 0
        return max(self.nr_update.values())

    def __call__(self, weights, gradient, key=None, lr_scale=1.):
        self.nr_update[key] += 1
        nr_upd = self.nr_update[key]
        lr = self.lr(nr_upd)
        lr *= lr_scale
        self.ops.clip_gradient(gradient, self.max_grad_norm)
        if key is None or self.mu == 0.0:
            weights -= lr * gradient
            gradient.fill(0)
        else:
            if key not in self.momentums:
                self.momentums[key] = self.ops.allocate(weights.size)
            momentum = self.momentums[key]
            if not self.nesterov:
                momentum *= self.mu
                momentum += gradient * lr
                weights -= momentum
            else:
                # http://cs231n.github.io/neural-networks-3/
                # v_prev = v # back this up
                # v = mu * v - lr * gradient # velocity update stays the same
                # x += -mu * v_prev + (1 + mu) * v # position update changes form
                # Implement this as
                # x += -mu * v
                # v *= mu
                # v -= lr * gradient
                # x += (1+mu) * v
                weights += -self.mu * momentum
                momentum *= self.mu
                momentum -= lr * gradient
                weights += (1+self.mu) * momentum
            gradient.fill(0)
        if self.averages is not None:
            if key not in self.averages:
                self.averages[key] = self.ops.allocate((weights.size,), dtype='float32')
            self.ops.update_averages(self.averages[key], weights, nr_upd)

    def lr(self, nr_upd):
        return linear_decay(self.alpha, self.decay, nr_upd)

    def set_loss(self, loss):
        pass


class Adam(SGD):
    def __init__(self, ops, lr, L2=1e-4, beta1=0.90, beta2=0.999, eps=1e-08, decay=0.0,
                 b1_decay=0.0, b2_decay=0.0, max_grad_norm=100.):
        self.ops = ops
        self.mom1 = {}
        self.mom2 = {}
        self.averages = {}
        self.nr_update = defaultdict(int)
        self.last_seen = defaultdict(int)
        self.max_grad_norm = max_grad_norm
        self.alpha = lr
        self.b1 = beta1
        self.b2 = beta2
        self.b1_decay = b1_decay
        self.b2_decay = b1_decay
        self.eps = eps
        self.decay = decay
        self.d = 1.
        self.f = 0.
        self.L2 = L2

    def lr(self, nr_upd):
        alpha = linear_decay(self.alpha, self.decay, nr_upd)
        fix1 = 1.- (self.b1 ** nr_upd)
        fix2 = 1.- (self.b2 ** nr_upd)
        return alpha * numpy.sqrt(fix2) / fix1

    def __call__(self, weights, gradient, lr_scale=1.,
            key=None):
        xp = get_array_module(weights)
        if xp is not self.ops.xp:
            if xp is numpy:
                self.ops = NumpyOps()
            else:
                self.ops = CupyOps()
        assert key is not None
        assert len(gradient) >= 1
        if key not in self.mom1:
            self.mom1[key] = self.ops.allocate(weights.size)
        if key not in self.mom2:
            self.mom2[key] = self.ops.allocate(weights.size)
        self.nr_update[key] += 1
        nr_upd = self.nr_update[key]
        if self.L2 != 0:
            gradient += self.L2 * weights
        if self.max_grad_norm:
            self.ops.clip_gradient(gradient, self.max_grad_norm)

        mom1 = self.mom1[key]
        mom2 = self.mom2[key]
        cdef weight_t lr = self.lr(nr_upd)
        cdef weight_t b1 = linear_decay(self.b1, self.b1_decay, nr_upd)
        cdef weight_t b2 = linear_decay(self.b2, self.b2_decay, nr_upd)
        cdef weight_t eps = self.eps
        self.ops.adam(
            weights, gradient, mom1, mom2, b1, b2, eps, lr * lr_scale)
        gradient.fill(0)
        if self.averages is not None:
            if key not in self.averages:
                self.averages[key] = self.ops.allocate((weights.size,), dtype='float32')
            self.ops.update_averages(self.averages[key], weights, nr_upd)

    def set_loss(self, loss):
        pass

class Eve(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.b3 = 0.9
        self.thl = 0.1
        self.thu = 10
        self.d = 1.
        self.loss_hat = None
        self.loss = None

    def __getattr__(self, attr):
        return getattr(self.optimizer, attr)

    def __call__(self, weights, gradient, key=None):
        assert self.d > 0.
        assert self.d < 100
        return self.optimizer(weights, gradient, key=key,
            lr_scale=1./self.d)

    def set_loss(self, loss):
        if self.loss_hat is None:
            self.loss_hat = loss
            self.loss = loss
            return
        prev_loss = self.loss
        prev_loss_hat = self.loss_hat
        loss_ch_fact = self._get_loss_ch_fact(loss, prev_loss_hat)

        loss_hat = (loss_ch_fact * prev_loss_hat)

        r = abs(loss_hat - prev_loss_hat) / min(loss_hat, prev_loss_hat)
        self.d = (self.b3 * self.d) + (1-self.b3) * r
        if self.d >= 100:
            print(self.d, locals())
            raise ValueError("Learning-rate adjustment exploded")
        self.loss_hat = loss_hat
        self.loss = loss

    def _get_loss_ch_fact(self, loss, loss_prev):
        lbound = (1+self.thl) if loss > loss_prev else (1/(1+self.thu))
        ubound = (1+self.thu) if loss > loss_prev else (1/(1+self.thl))
        return min(ubound, max(lbound, loss / loss_prev))


