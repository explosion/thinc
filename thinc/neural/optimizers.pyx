# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libc.stdlib cimport calloc, malloc, free
import math

from collections import defaultdict
import numpy

from ..typedefs cimport weight_t
from .ops import NumpyOps, CupyOps, add_gradient_noise
from .util import get_array_module


def linear_decay(rate, decay, nr_upd):
    return rate * 1./(1. + decay * nr_upd)


def anneal(rate, decay, decay_steps, nr_upd):
    if decay == 0.0:
        return rate
    else:
        return rate * decay ** (nr_upd / decay_steps)

def Adam(*args, **kwargs):
    return Optimizer(*args, **kwargs)


def SGD(*args, **kwargs):
    kwargs.setdefault('beta1', 0.)
    kwargs.setdefault('beta2', 0.)
    return Optimizer(*args, **kwargs)


class Optimizer(object):
    '''Do various flavours of stochastic gradient descent, with first and
    second order momentum.
    
    Examples
    
    * beta1=0., beta2=0.: "vanilla" SGD
    * beta1=0.9, beta2=0.: "Classic momentum"
    * beta1=0.0, beta2=0.2: RMS prop
    * b1=0.999, b2=0.9: Adam
    '''
    def __init__(self, ops, lr, L2=1e-4, beta1=0.90, beta2=0.999, eps=1e-08, decay=0.0,
                 decay_steps=5000,
                 b1_decay=0.0, b2_decay=0.0, max_grad_norm=10., gradient_noise=0.0,
                 nesterov=True, L2_is_weight_decay=False, lookahead_k=0,
                 lookahead_alpha=0.5, use_radam=False, use_lars=False):
        self.ops = ops
        self.mom1 = {}
        self.mom2 = {}
        self.slow_weights = {} # For lookahead
        self.averages = {}
        self.nr_update = defaultdict(int)
        self.last_seen = defaultdict(int)
        self.max_grad_norm = max_grad_norm
        self.alpha = lr
        self.b1 = beta1
        self.b2 = beta2
        self.b1_decay = b1_decay
        self.b2_decay = b1_decay
        self.gradient_noise = gradient_noise
        self.eps = eps
        self.decay = decay
        self.L2 = L2
        self.nesterov = nesterov
        self.decay_steps = decay_steps
        self.L2_is_weight_decay = L2_is_weight_decay
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
        self.use_radam = use_radam
        self.use_lars = use_lars
        self.lars_min = 0
        self.lars_max = 10

    def to_gpu(self):
        self.ops = CupyOps()
        for params in (self.mom1, self.mom2, self.averages):
            for key, value in params.items():
                params[key] = self.ops.xp.asarray(value, dtype=value.dtype)
    
    def to_cpu(self):
        self.ops = NumpyOps()
        for params in (self.mom1, self.mom2, self.averages):
            for key, value in params.items():
                if hasattr(value, 'get'):
                    params[key] = value.get()

    def lr(self, nr_upd):
        alpha = anneal(self.alpha, self.decay, self.decay_steps, nr_upd)
        if self.b1 == 0. or self.b2 == 0.:
            return alpha
        fix1 = 1.- (self.b1 ** nr_upd)
        fix2 = 1.- (self.b2 ** nr_upd)
        return alpha * numpy.sqrt(fix2) / fix1

    def __call__(self, weights, gradient, lr_scale=1., key=None):
        assert len(gradient) >= 1
        xp = get_array_module(weights)
        if xp is not self.ops.xp:
            if xp is numpy:
                self.ops = NumpyOps()
            else:
                self.ops = CupyOps()

        self.nr_update[key] += 1
        nr_upd = self.nr_update[key]
        if self.L2 != 0 and not self.L2_is_weight_decay:
            gradient += self.L2 * weights
        if self.max_grad_norm:
            self.ops.clip_gradient(gradient, self.max_grad_norm)
        if self.gradient_noise:
            add_gradient_noise(gradient, self.gradient_noise, nr_upd)
        if self.use_radam:
            self._radam(xp, weights, gradient, lr_scale, key, nr_upd)
        elif self.b1 > 0. and self.b2 > 0.:
            self._adam(xp, weights, gradient, lr_scale, key, nr_upd)
        elif self.b1 > 0. and not self.nesterov:
            raise NotImplementedError
        elif self.b1 > 0.:
            self._nesterov(xp, weights, gradient, lr_scale, key)
        elif self.b2 > 0.:
            raise NotImplementedError
        else:
            weights -= lr_scale * self.alpha * gradient
        gradient.fill(0.)
        if self.L2 != 0 and self.L2_is_weight_decay:
            weights -= self.L2 * weights
        if self.lookahead_k and self.nr_update[key] % self.lookahead_k == 0:
            if key not in self.slow_weights:
                self.slow_weights[key] = self.ops.allocate((weights.size,), dtype='float32')
            slow = self.slow_weights[key]
            slow += self.lookahead_alpha * (weights - slow)
            weights[:] = slow
        if self.averages is not None:
            if key not in self.averages:
                self.averages[key] = self.ops.allocate((weights.size,), dtype='float32')
            self.ops.update_averages(self.averages[key], weights, nr_upd)

    def _radam(self, xp, weights, gradient, lr_scale, key, nr_upd):
        if key not in self.mom1:
            self.mom1[key] = self.ops.allocate(weights.size)
        if key not in self.mom2:
            self.mom2[key] = self.ops.allocate(weights.size)
 
        beta1 = self.b1
        beta2 = self.b2
        eps = self.eps
        sma_inf = 2 / (1-beta2) - 1

        exp_avg = self.mom1[key]
        exp_avg_sq = self.mom2[key]
        # Decay the first and second moment running average coefficient
        exp_avg *= beta1
        exp_avg += (1-beta1) * gradient
        exp_avg_sq *= beta2
        exp_avg_sq += (1-beta2) * gradient**2
        # Bias correction
        bias_correction1 = 1 - beta1 ** nr_upd
        bias_correction2 = 1 - beta2 ** nr_upd

        # Compute length of SMA
        sma_t = sma_inf - 2 * nr_upd * (1 - bias_correction2) / bias_correction2
        update = self.ops.allocate(weights.shape, dtype="f")
        if sma_t > 4:
            # Variance rectification term
            r_t = math.sqrt((sma_t - 4) * (sma_t - 2) * sma_inf / ((sma_inf - 4) * (sma_inf - 2) * sma_t))
            # Adaptive momentum
            update += r_t * (
                (exp_avg / bias_correction1)
                /
                (self.ops.xp.sqrt(exp_avg_sq / bias_correction2) + eps)
            )
        else:
            # Unadapted momentum
            update += exp_avg / bias_correction1
        if self.use_lars:
            # LARS
            w_norm = self.ops.xp.linalg.norm(weights)
            u_norm = self.ops.xp.linalg.norm(update)
            phi_p = min(max(w_norm, self.lars_min), self.lars_max)
            # Compute the local LR
            if phi_p == 0 or u_norm == 0:
                local_lr = 1
            else:
                local_lr = phi_p / u_norm
            lr = self.alpha * lr_scale * local_lr
        else:
            lr = self.alpha * lr_scale
        weights -= lr * update

    def _nesterov(self, xp, weights, gradient, lr_scale, key):
        # http://cs231n.github.io/neural-networks-3/
        # v_prev = v # back this up
        # v = mu * v - lr * gradient # velocity update stays the same
        # x += -mu * v_prev + (1 + mu) * v # position update changes form
        # Implement this as
        # x += -mu * v
        # v *= mu
        # v -= lr * gradient
        # x += (1+mu) * v
        lr = self.alpha * lr_scale
        if key not in self.mom1:
            self.mom1[key] = self.ops.allocate(weights.size)
        momentum = self.mom1[key]
        weights += -self.b1 * momentum
        momentum *= self.b1
        momentum -= lr * gradient
        weights += (1+self.b1) * momentum

    def _adam(self, xp, weights, gradient, lr_scale, key, nr_upd):
        if key not in self.mom1:
            self.mom1[key] = self.ops.allocate(weights.size)
        if key not in self.mom2:
            self.mom2[key] = self.ops.allocate(weights.size)
        mom1 = self.mom1[key]
        mom2 = self.mom2[key]
        cdef weight_t lr = self.lr(nr_upd)
        cdef weight_t b1 = linear_decay(self.b1, self.b1_decay, nr_upd)
        cdef weight_t b2 = linear_decay(self.b2, self.b2_decay, nr_upd)
        cdef weight_t eps = self.eps
        self.ops.adam(
            weights, gradient, mom1, mom2, b1, b2, eps, lr * lr_scale)
        gradient.fill(0)


