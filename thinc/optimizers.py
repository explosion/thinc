import math

from typing import Dict, Optional, Union, Tuple, List, cast
from collections import defaultdict
import numpy

from .backends import Ops, NumpyOps, CupyOps, get_current_ops
from .types import Array, Generator
from .util import get_array_module
from .config import registry


# We need to use the custom Generator type for schedules to work around pydantic
# not supporting Iterator / Iterable
ScheduleT = Generator

KeyT = Tuple[int, str]
FloatOrSeq = Union[float, List[float], Generator]
IntOrSeq = Union[int, List[int], Generator]

SGD_DEFAULTS: Dict[str, Union[float, bool, int]] = {
    "L2": 1e-6,
    "L2_is_weight_decay": True,
    "grad_clip": 1.0,
}


ADAM_DEFAULTS: Dict[str, Union[float, bool, int]] = {
    "learn_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-08,
    "L2": SGD_DEFAULTS["L2"],
    "grad_clip": SGD_DEFAULTS["grad_clip"],
    "L2_is_weight_decay": True,
}


@registry.optimizers("RAdam.v1")
def RAdam(
    learn_rate: FloatOrSeq = ADAM_DEFAULTS["learn_rate"],
    *,
    beta1: FloatOrSeq = ADAM_DEFAULTS["beta1"],
    beta2: FloatOrSeq = ADAM_DEFAULTS["beta2"],
    eps: FloatOrSeq = ADAM_DEFAULTS["eps"],
    weight_decay: FloatOrSeq = ADAM_DEFAULTS["L2"],
    grad_clip: FloatOrSeq = ADAM_DEFAULTS["grad_clip"],
    lookahead_k: int = 0,
    lookahead_alpha: FloatOrSeq = 0.5,
    use_averages: bool = True,
    ops: Optional[Ops] = None
):
    return Optimizer(
        learn_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        grad_clip=grad_clip,
        L2_is_weight_decay=True,
        L2=weight_decay,
        lookahead_k=lookahead_k,
        lookahead_alpha=lookahead_alpha,
        use_averages=True,
        use_radam=True,
        ops=ops,
    )


@registry.optimizers("Adam.v1")
def Adam(
    learn_rate: FloatOrSeq = ADAM_DEFAULTS["learn_rate"],
    *,
    L2: FloatOrSeq = ADAM_DEFAULTS["L2"],
    beta1: FloatOrSeq = ADAM_DEFAULTS["beta1"],
    beta2: FloatOrSeq = ADAM_DEFAULTS["beta2"],
    eps: FloatOrSeq = ADAM_DEFAULTS["eps"],
    grad_clip: FloatOrSeq = ADAM_DEFAULTS["grad_clip"],
    L2_is_weight_decay: bool = cast(bool, ADAM_DEFAULTS["L2_is_weight_decay"]),
    use_averages: bool = True,
    lookahead_k: int = 0,
    lookahead_alpha: FloatOrSeq = 0.5,
    ops: Optional[Ops] = None
):
    return Optimizer(
        learn_rate,
        L2=L2,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        grad_clip=grad_clip,
        L2_is_weight_decay=L2_is_weight_decay,
        use_averages=True,
        lookahead_k=lookahead_k,
        lookahead_alpha=lookahead_alpha,
        use_radam=False,
        ops=ops,
    )


@registry.optimizers("SGD.v1")
def SGD(
    learn_rate: FloatOrSeq,
    *,
    ops: Optional[Ops] = None,
    L2: FloatOrSeq = SGD_DEFAULTS["L2"],
    grad_clip: FloatOrSeq = SGD_DEFAULTS["grad_clip"],
    L2_is_weight_decay: bool = cast(bool, SGD_DEFAULTS["L2_is_weight_decay"]),
    use_averages: bool = True
):
    return Optimizer(
        learn_rate,
        L2=L2,
        grad_clip=grad_clip,
        L2_is_weight_decay=L2_is_weight_decay,
        beta1=0.0,
        beta2=0.0,
        ops=ops,
    )


class Optimizer(object):
    """Do various flavours of stochastic gradient descent, with first and
    second order momentum. Currently support 'vanilla' SGD, Adam, and RAdam.
    """

    mom1: Dict[KeyT, Array]
    mom2: Dict[KeyT, Array]
    slow_weights: Dict[KeyT, Array]
    averages: Optional[Dict[KeyT, Array]]
    schedules: Dict[str, Generator]
    nr_update: Dict[KeyT, int]
    last_seen: Dict[KeyT, int]
    grad_clip: float
    alpha: float
    b1: float
    b2: float
    eps: float
    L2: float
    lookahead_k: int
    lookahead_alpha: float
    use_radam: bool
    L2_is_weight_decay: bool
    _radam_buffer: List[List[Optional[Array]]]

    def __init__(
        self,
        learn_rate: FloatOrSeq,
        *,
        ops: Optional[Ops] = None,
        L2: FloatOrSeq = ADAM_DEFAULTS["L2"],
        beta1: FloatOrSeq = ADAM_DEFAULTS["beta1"],
        beta2: FloatOrSeq = ADAM_DEFAULTS["beta2"],
        eps: FloatOrSeq = ADAM_DEFAULTS["eps"],
        grad_clip: FloatOrSeq = ADAM_DEFAULTS["grad_clip"],
        lookahead_k: int = 0,
        lookahead_alpha: FloatOrSeq = 0.5,
        use_averages: bool = True,
        use_radam: bool = False,
        L2_is_weight_decay: bool = True,
    ):
        """
        Initialize an optimizer.

        learn_rate (float): The initial learning rate.
        ops (Ops): A backend object. Defaults to the currently selected backend.
        L2 (float): The L2 regularization term.
        beta1 (float): First-order momentum.
        beta2 (float): Second-order momentum.
        eps (float): Epsilon term for Adam etc.
        grad_clip (float): Gradient clipping.
        lookahead_k (int): K parameter for lookahead.
        lookahead_alpha (float): Alpha parameter for lookahead.
        use_averages (bool): Whether to track moving averages of the parameters.
        use_radam (bool): Whether to use the RAdam optimizer.
        L2_is_weight_decay (bool): Whether to interpret the L2 parameter as a
            weight decay term, in the style of the AdamW optimizer.
        """
        self.ops = ops if ops is not None else get_current_ops()
        self.mom1 = {}
        self.mom2 = {}
        self.slow_weights = {}  # For lookahead
        if use_averages:
            self.averages = {}
        else:
            self.averages = None
        self.schedules = {}
        self.nr_update = defaultdict(int)
        self.last_seen = defaultdict(int)
        self._set_attr_or_schedule("grad_clip", grad_clip)
        self._set_attr_or_schedule("learn_rate", learn_rate)
        self._set_attr_or_schedule("b1", beta1)
        self._set_attr_or_schedule("b2", beta2)
        self._set_attr_or_schedule("eps", eps)
        self._set_attr_or_schedule("L2", L2)
        self._set_attr_or_schedule("lookahead_alpha", lookahead_alpha)
        self._set_attr_or_schedule("lookahead_k", lookahead_k)
        self.use_radam = use_radam
        self.L2_is_weight_decay = L2_is_weight_decay
        self._radam_buffer = [[None, None, None] for _ in range(10)]

    def _set_attr_or_schedule(self, name, value):
        if isinstance(value, (float, bool, int)):
            setattr(self, name, value)
        else:
            if isinstance(value, list):
                value = iter(value)
            self.schedules[name] = value
            setattr(self, name, next(value))

    def to_gpu(self):
        self.ops = CupyOps()
        for params in (self.mom1, self.mom2, self.averages):
            for key, value in params.items():
                params[key] = self.ops.xp.asarray(value, dtype=value.dtype)

    def to_cpu(self):
        self.ops = NumpyOps()
        for params in (self.mom1, self.mom2, self.averages):
            for key, value in params.items():
                if hasattr(value, "get"):
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

    def __call__(self, weights: Array, gradient: Array, *, lr_scale: float = 1.0, key):
        if len(gradient) < 1:
            return weights, gradient
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
        if self.grad_clip:
            self.ops.clip_gradient(gradient, self.grad_clip)
        if self.use_radam:
            self._radam(xp, weights, gradient, lr_scale, key, nr_upd)
        elif self.b1 > 0.0 and self.b2 > 0.0:
            self._adam(xp, weights, gradient, lr_scale, key, nr_upd)
        elif self.b2 > 0.0:
            raise NotImplementedError  # TODO: error message
        else:
            weights -= lr_scale * self.alpha * gradient
        gradient.fill(0.0)
        if self.L2 != 0 and self.L2_is_weight_decay:
            weights -= self.L2 * weights
        if self.lookahead_k and self.nr_update[key] % self.lookahead_k == 0:
            if key not in self.slow_weights:
                self.slow_weights[key] = self.ops.alloc_f1d(
                    weights.size, dtype="float32"
                )
            slow = self.slow_weights[key]
            slow += self.lookahead_alpha * (weights - slow)
            weights[:] = slow
        if self.averages is not None:
            if key not in self.averages:
                self.averages[key] = self.ops.alloc(weights.shape, dtype="float32")
            self.ops.update_averages(self.averages[key], weights, nr_upd)
        return weights, gradient

    def _radam(self, xp, weights, grad, lr_scale, key, nr_upd):
        if key not in self.mom1:
            self.mom1[key] = self.ops.alloc_f1d(weights.size)
        if key not in self.mom2:
            self.mom2[key] = self.ops.alloc_f1d(weights.size)

        # While we port from PyTorch
        p_data_fp32 = weights
        state = {
            "step": self.nr_update[key],
            "exp_avg": self.mom1[key],
            "exp_avg_sq": self.mom2[key],
        }
        group = {
            "lr": self.alpha,
            "betas": [self.b1, self.b2],
            "eps": self.eps,
            "weight_decay": 0.0,
            "buffer": self._radam_buffer,
        }
        degenerated_to_sgd = True

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]

        # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        exp_avg_sq *= beta2
        exp_avg_sq += (1 - beta2) * (grad ** 2)
        # exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg *= beta1
        exp_avg += (1 - beta1) * grad

        state["step"] += 1
        buffered = group["buffer"][int(state["step"] % 10)]
        if state["step"] == buffered[0]:
            N_sma, step_size = buffered[1], buffered[2]
        else:
            buffered[0] = state["step"]
            beta2_t = beta2 ** state["step"]
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
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
                step_size = 1.0 / (1 - beta1 ** state["step"])
            else:
                step_size = -1
            buffered[2] = step_size

        # more conservative since it's an approximated value
        if N_sma >= 5:
            if group["weight_decay"] != 0:
                p_data_fp32 += -group["weight_decay"] * group["lr"] * p_data_fp32
            denom = xp.sqrt(exp_avg_sq) + group["eps"]
            p_data_fp32 += -step_size * group["lr"] * (exp_avg / denom)
        elif step_size > 0:
            if group["weight_decay"] != 0:
                p_data_fp32 += -group["weight_decay"] * group["lr"] * p_data_fp32
            p_data_fp32 += -step_size * group["lr"] * exp_avg

    def _lookahead(self, weights, key):
        if self.lookahead_k and self.nr_update[key] % self.lookahead_k == 0:
            if key not in self.slow_weights:
                self.slow_weights[key] = self.ops.alloc_f1d(
                    weights.size, dtype="float32"
                )
            slow = self.slow_weights[key]
            slow += self.lookahead_alpha * (weights - slow)
            weights[:] = slow

    def _adam(self, xp, weights, gradient, lr_scale, key, nr_upd):
        weights_1D = weights.reshape((weights.size, ))
        gradient_1D = gradient.reshape((gradient.size, ))
        if key not in self.mom1:
            self.mom1[key] = self.ops.alloc_f1d(weights.size)
        if key not in self.mom2:
            self.mom2[key] = self.ops.alloc_f1d(weights.size)
        mom1 = self.mom1[key]
        mom2 = self.mom2[key]
        fix1 = 1.0 - (self.b1 ** nr_upd)
        fix2 = 1.0 - (self.b2 ** nr_upd)
        lr = self.learn_rate * numpy.sqrt(fix2) / fix1
        b1 = self.b1
        b2 = self.b2
        eps = self.eps
        self.ops.adam(weights_1D, gradient_1D, mom1, mom2, b1, b2, eps, lr * lr_scale)


__all__ = ["Adam", "RAdam", "SGD", "Optimizer", "ADAM_DEFAULTS", "SGD_DEFAULTS"]
