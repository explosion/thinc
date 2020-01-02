# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libc.stdlib cimport calloc, malloc, free
import math

from typing import Sequence, Dict, Optional, Union, Any
from collections import defaultdict
import numpy

from .backends import Ops, NumpyOps, CupyOps, get_current_ops
from .types import Array
from .util import get_array_module
from ._registry import registry


ctypedef float weight_t


SGD_DEFAULTS = {
    "L2": 1e-4,
    "L2_is_weight_decay": True,
    "max_grad_norm": 10,
}


ADAM_DEFAULTS = {
    "learn_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-08,
    "L2": SGD_DEFAULTS["L2"],
    "max_grad_norm": SGD_DEFAULTS["max_grad_norm"],
    "L2_is_weight_decay": True,
    "schedules": None,
}


@registry.optimizers.register("RAdam.v1")
def RAdam(
        learn_rate: float = ADAM_DEFAULTS["learn_rate"],
        beta1: float = ADAM_DEFAULTS["beta1"],
        beta2: float = ADAM_DEFAULTS["beta2"],
        eps: float = ADAM_DEFAULTS["eps"],
        weight_decay: float = ADAM_DEFAULTS["L2"],
        max_grad_norm: float = ADAM_DEFAULTS["max_grad_norm"],
        lookahead_k: int = 0,
        lookahead_alpha: float = 0.5,
        use_averages: bool = True,
        schedules: Dict[str, Sequence[float]] = None,
        ops: Optional[Ops] = None,
):
    return Optimizer(
        learn_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        max_grad_norm=max_grad_norm,
        L2_is_weight_decay=True,
        L2=weight_decay,
        schedules=schedules,
        lookahead_k=lookahead_k,
        lookahead_alpha=lookahead_alpha,
        use_averages=True,
        use_radam=True, use_lars=False, ops=ops
    )


@registry.optimizers.register("Adam.v1")
def Adam(
        learn_rate: float = ADAM_DEFAULTS["learn_rate"],
        L2: float = ADAM_DEFAULTS["L2"],
        beta1: float = ADAM_DEFAULTS["beta1"],
        beta2: float = ADAM_DEFAULTS["beta2"],
        eps: float = ADAM_DEFAULTS["eps"],
        max_grad_norm: float = ADAM_DEFAULTS["max_grad_norm"],
        L2_is_weight_decay: bool = ADAM_DEFAULTS["L2_is_weight_decay"],
        use_averages: bool = True,
        lookahead_k: int = 0,
        lookahead_alpha: float = 0.5,
        ops: Optional[Ops] = None,
        schedules: Optional[Dict[str, Sequence[float]]] = None,
):
    return Optimizer(
        learn_rate,
        L2=L2,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        max_grad_norm=max_grad_norm,
        L2_is_weight_decay=L2_is_weight_decay,
        schedules=schedules,
        use_averages=True,
        decay_steps=0,
        lookahead_k=lookahead_k,
        lookahead_alpha=lookahead_alpha,
        use_radam=False,
        use_lars=False,
        ops=ops
    )


@registry.optimizers.register("SGD.v1")
def SGD(
        learn_rate: float,
        ops: Optional[Ops] = None,
        L2: float = SGD_DEFAULTS["L2"],
        max_grad_norm: float = SGD_DEFAULTS["max_grad_norm"],
        L2_is_weight_decay: bool = SGD_DEFAULTS["L2_is_weight_decay"],
        use_averages: bool = True,
        schedules: Optional[Dict[str, Sequence[float]]] = None,
):
    return Optimizer(
        learn_rate,
        L2=L2,
        max_grad_norm=max_grad_norm,
        L2_is_weight_decay=L2_is_weight_decay,
        schedules=schedules,
        beta1=0.0,
        beta2=0.0,
        ops=ops
    )


class Optimizer(object):
    """Do various flavours of stochastic gradient descent, with first and
    second order momentum.

    Examples:
    * beta1=0., beta2=0.: "vanilla" SGD
    * beta1=0.9, beta2=0.: "Classic momentum"
    * beta1=0.0, beta2=0.2: RMS prop
    * b1=0.999, b2=0.9: Adam
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        return registry.make_from_config(config)

    def __init__(
        self,
        lr: float,
        *,
        ops: Optional[Ops] = None,
        L2: float = 1e-4,
        beta1: float = 0.90,
        beta2: float = 0.999,
        eps: float = 1e-08,
        max_grad_norm: float = 10.0,
        lookahead_k: int = 0,
        lookahead_alpha: float = 0.5,
        use_averages: bool = True,
        use_radam: bool = False,
        L2_is_weight_decay: bool = True,
        schedules: Optional[Dict[str, Sequence[float]]] = None,
        **_,
    ):
        self.ops = ops if ops is not None else get_current_ops()
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
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
        self.use_radam = use_radam
        self.L2_is_weight_decay = L2_is_weight_decay
        self._radam_buffer = [[None, None, None] for _ in range(10)]

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
    def learn_rate(self) -> float:
        return self.alpha

    @learn_rate.setter
    def learn_rate(self, learn_rate):
        self.alpha = learn_rate

    def __call__(self, weights, gradient: Array, lr_scale: float = 1.0, key=None):
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

    def _radam(self, xp, weights, grad, lr_scale, key, nr_upd):
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
        exp_avg_sq += (1-beta2) * (grad ** 2)
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
                step_size = math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            ) / (1 - beta1 ** state["step"])
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

    def _adam(self, xp, weights, gradient, lr_scale, key, nr_upd):
        if key not in self.mom1:
            self.mom1[key] = self.ops.allocate(weights.size)
        if key not in self.mom2:
            self.mom2[key] = self.ops.allocate(weights.size)
        mom1 = self.mom1[key]
        mom2 = self.mom2[key]
        fix1 = 1.- (self.b1 ** nr_upd)
        fix2 = 1.- (self.b2 ** nr_upd)
        cdef weight_t lr = self.learn_rate * numpy.sqrt(fix2) / fix1
        cdef weight_t b1 = self.b1
        cdef weight_t b2 = self.b2
        cdef weight_t eps = self.eps
        self.ops.adam(
            weights, gradient, mom1, mom2, b1, b2, eps, lr * lr_scale)
