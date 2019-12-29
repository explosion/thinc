# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libc.stdlib cimport calloc, malloc, free
import math

from typing import Sequence, Dict
from collections import defaultdict
import numpy

from .backends import NumpyOps, CupyOps
from .util import get_array_module
from ._registry import registry

ctypedef float weight_t

SGD_DEFAULTS = {
    "L2": 1e-4,
    "max_grad_norm": 10,
    "L2_is_weight_decay": False
}


ADAM_DEFAULTS = {
    "learn_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-08,
    "L2": SGD_DEFAULTS["L2"],
    "max_grad_norm": SGD_DEFAULTS["max_grad_norm"],
    "L2_is_weight_decay": SGD_DEFAULTS["L2_is_weight_decay"],
    "schedules": None
}


@registry.optimizers.register("RAdam.v1")
def create_RAdam(
        learn_rate: float=ADAM_DEFAULTS["learn_rate"],
        beta1: float=ADAM_DEFAULTS["beta1"],
        beta2: float=ADAM_DEFAULTS["beta2"],
        eps: float=ADAM_DEFAULTS["eps"],
        weight_decay: float=ADAM_DEFAULTS["L2"],
        max_grad_norm: float=ADAM_DEFAULTS["max_grad_norm"],
        lookahead_k: int=0,
        lookahead_alpha: float=0.5,
        use_averages: bool=True,
        schedules: Dict[str, Sequence[float]]=None,
        ops=None,
):
    ops = _make_ops(ops)
    return Optimizer(
        ops,
        learn_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        max_grad_norm=max_grad_norm,
        L2_is_weight_decay=True,
        L2=weight_decay,
        schedules=schedules,
        nesterov=None, lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha,
        use_averages=True,
        use_radam=True, use_lars=False
    )


@registry.optimizers.register("Adam.v1")
def create_Adam(
        learn_rate: float=ADAM_DEFAULTS["learn_rate"],
        L2: float=ADAM_DEFAULTS["L2"],
        beta1: float=ADAM_DEFAULTS["beta1"],
        beta2: float=ADAM_DEFAULTS["beta2"],
        eps: float=ADAM_DEFAULTS["eps"],
        max_grad_norm: float=ADAM_DEFAULTS["max_grad_norm"],
        L2_is_weight_decay: bool=ADAM_DEFAULTS["L2_is_weight_decay"],
        use_averages: bool=True,
        lookahead_k: int=0,
        lookahead_alpha: float=0.5,
        ops=None,
        schedules: Dict[str, Sequence[float]]=None
):
    ops = _make_ops(ops)
    return Optimizer(
        ops,
        learn_rate,
        L2=L2,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        max_grad_norm=max_grad_norm,
        L2_is_weight_decay=L2_is_weight_decay,
        schedules=schedules,
        use_averages=True,
        decay=0.0, decay_steps=0, b1_decay=0, b2_decay=0,
        nesterov=None, lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha,
        use_radam=False, use_lars=False
    )


@registry.optimizers.register("SGD.v1")
def create_SGD(learn_rate,
        ops=None,
        L2=SGD_DEFAULTS["L2"],
        max_grad_norm=SGD_DEFAULTS["max_grad_norm"],
        L2_is_weight_decay=SGD_DEFAULTS["L2_is_weight_decay"],
        use_averages=True,
        schedules=None
):
    ops = _make_ops(ops)
    return Optimizer(ops, learn_rate,
        L2=L2, max_grad_norm=max_grad_norm, L2_is_weight_decay=L2_is_weight_decay,
        schedules=schedules, beta1=0.0, beta2=0.0)


class Optimizer(object):
    '''Do various flavours of stochastic gradient descent, with first and
    second order momentum.

    Examples

    * beta1=0., beta2=0.: "vanilla" SGD
    * beta1=0.9, beta2=0.: "Classic momentum"
    * beta1=0.0, beta2=0.2: RMS prop
    * b1=0.999, b2=0.9: Adam
    '''
    @classmethod
    def from_config(cls, config):
        return registry.make_from_config(config)

    def __init__(self, ops, lr, L2=1e-4, beta1=0.90, beta2=0.999, eps=1e-08,
                 max_grad_norm=10., nesterov=True,
                 L2_is_weight_decay=False, lookahead_k=0, lookahead_alpha=0.5,
                 use_averages=True, use_radam=False, use_lars=False, schedules=None, **_):
        self.ops = ops
        if schedules is None:
            self.schedules = {}
        else:
            self.schedules = dict(schedules)
        self.mom1 = {}
        self.mom2 = {}
        self.slow_weights = {} # For lookahead
        if use_averages:
            self.averages = {}
        else:
            self.averages = None
        self.nr_update = defaultdict(int)
        self.last_seen = defaultdict(int)
        self.max_grad_norm = max_grad_norm
        self.alpha = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.L2 = L2
        self.nesterov = nesterov
        self.L2_is_weight_decay = L2_is_weight_decay
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
        self.use_radam = use_radam
        self._radam_buffer = [[None, None, None] for _ in range(10)]
        # Deprecated
        self.use_lars = use_lars
        self.decay = 0.0
        self.decay_steps = 0
        self.lars_min = 0
        self.lars_max = 10
        self.b1_decay = 0.0
        self.b2_decay = 0.0

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

    def step_schedules(self):
        for key, schedule in self.schedules.items():
            setattr(self, key, next(schedule))

    @property
    def learn_rate(self):
        return self.alpha

    @learn_rate.setter
    def learn_rate(self, learn_rate):
        self.alpha = learn_rate

    def lr(self, nr_upd):
        alpha = anneal(self.alpha, self.decay, self.decay_steps, nr_upd)
        if self.b1 == 0. or self.b2 == 0.:
            return alpha
        fix1 = 1.- (self.b1 ** nr_upd)
        fix2 = 1.- (self.b2 ** nr_upd)
        return alpha * numpy.sqrt(fix2) / fix1

    def __call__(self, weights, gradient, lr_scale=1., key=None):
        if len(gradient) < 1:
            return
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
        if self.use_radam:
            self._radam2(xp, weights, gradient, lr_scale, key, nr_upd)
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
        self._lookahead(weights, key)

    def _radam2(self, xp, weights, grad, lr_scale, key, nr_upd):
        if key not in self.mom1:
            self.mom1[key] = self.ops.allocate(weights.size)
        if key not in self.mom2:
            self.mom2[key] = self.ops.allocate(weights.size)

        # While we port from PyTorch
        p_data_fp32 = weights
        state = {
            "step": self.nr_update[key],
            "exp_avg": self.mom1[key],
            "exp_avg_sq": self.mom2[key]
        }
        group = {
            "lr": self.alpha,
            "betas": [self.b1, self.b2],
            "eps": self.eps,
            "weight_decay": 0.0,
            "buffer": self._radam_buffer
        }
        degenerated_to_sgd = True

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        exp_avg_sq *= beta2
        exp_avg_sq += (1-beta2) * (grad * grad)
        # exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg *= beta1
        exp_avg += (1-beta1) * grad

        state['step'] += 1
        buffered = group['buffer'][int(state['step'] % 10)]
        if state['step'] == buffered[0]:
            N_sma, step_size = buffered[1], buffered[2]
        else:
            buffered[0] = state['step']
            beta2_t = beta2 ** state['step']
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
            buffered[1] = N_sma

            # more conservative since it's an approximated value
            if N_sma >= 5:
                step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
            elif degenerated_to_sgd:
                step_size = 1.0 / (1 - beta1 ** state['step'])
            else:
                step_size = -1
            buffered[2] = step_size

        # more conservative since it's an approximated value
        if N_sma >= 5:
            if group['weight_decay'] != 0:
                p_data_fp32 += -group["weight_decay"] * group["lr"] * p_data_fp32
            denom = xp.sqrt(exp_avg_sq) + group['eps']
            p_data_fp32 += -step_size * group["lr"] * (exp_avg / denom)
        elif step_size > 0:
            if group['weight_decay'] != 0:
                p_data_fp32 += -group["weight_decay"] * group["lr"] * p_data_fp32
            p_data_fp32 += -step_size * group["lr"] * exp_avg
        self._lookahead(weights, key)

    def _lookahead(self, weights, key):
        if self.lookahead_k and self.nr_update[key] % self.lookahead_k == 0:
            if key not in self.slow_weights:
                self.slow_weights[key] = self.ops.allocate((weights.size,), dtype='float32')
            slow = self.slow_weights[key]
            slow += self.lookahead_alpha * (weights - slow)
            weights[:] = slow


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


def _make_ops(ops):
    if ops == "CupyOps":
        return CupyOps()
    elif ops == "NumpyOps":
        return NumpyOps()
    elif ops is None:
        from ._classes.model import Model
        return Model.ops
    else:
        return ops


# These are deprecated

def Adam(*args, **kwargs):
    return Optimizer(*args, **kwargs)

def SGD(*args, **kwargs):
    kwargs.setdefault('beta1', 0.)
    kwargs.setdefault('beta2', 0.)
    return Optimizer(*args, **kwargs)


def linear_decay(rate, decay, nr_upd):
    return rate * 1./(1. + decay * nr_upd)


def anneal(rate, decay, decay_steps, nr_upd):
    if decay == 0.0:
        return rate
    else:
        return rate * decay ** (nr_upd / decay_steps)
