from typing import Any, Dict, Optional, Union, Tuple, List, cast
from collections import defaultdict
import itertools
import math
from types import GeneratorType

from .backends import get_array_ops
from .types import Generator, FloatsXd
from .config import registry
from .schedules import constant, Schedule


KeyT = Tuple[int, str]
ScheduleT = Union[float, List[float], Generator, Schedule]

SGD_DEFAULTS: Dict[str, Union[float, bool, int]] = {
    "L2": 0.0,
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
    learn_rate: ScheduleT = ADAM_DEFAULTS["learn_rate"],
    *,
    beta1: ScheduleT = ADAM_DEFAULTS["beta1"],
    beta2: ScheduleT = ADAM_DEFAULTS["beta2"],
    eps: ScheduleT = ADAM_DEFAULTS["eps"],
    L2: ScheduleT = ADAM_DEFAULTS["L2"],
    L2_is_weight_decay: bool = cast(bool, ADAM_DEFAULTS["L2_is_weight_decay"]),
    grad_clip: ScheduleT = ADAM_DEFAULTS["grad_clip"],
    use_averages: bool = True,
):
    return Optimizer(
        learn_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        grad_clip=grad_clip,
        L2_is_weight_decay=L2_is_weight_decay,
        L2=L2,
        use_averages=use_averages,
        use_radam=True,
    )


@registry.optimizers("Adam.v1")
def Adam(
    learn_rate: ScheduleT = ADAM_DEFAULTS["learn_rate"],
    *,
    L2: ScheduleT = ADAM_DEFAULTS["L2"],
    beta1: ScheduleT = ADAM_DEFAULTS["beta1"],
    beta2: ScheduleT = ADAM_DEFAULTS["beta2"],
    eps: ScheduleT = ADAM_DEFAULTS["eps"],
    grad_clip: ScheduleT = ADAM_DEFAULTS["grad_clip"],
    L2_is_weight_decay: bool = cast(bool, ADAM_DEFAULTS["L2_is_weight_decay"]),
    use_averages: bool = True,
):
    return Optimizer(
        learn_rate,
        L2=L2,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        grad_clip=grad_clip,
        L2_is_weight_decay=L2_is_weight_decay,
        use_averages=use_averages,
        use_radam=False,
    )


@registry.optimizers("SGD.v1")
def SGD(
    learn_rate: ScheduleT,
    *,
    L2: ScheduleT = SGD_DEFAULTS["L2"],
    grad_clip: ScheduleT = SGD_DEFAULTS["grad_clip"],
    L2_is_weight_decay: bool = cast(bool, SGD_DEFAULTS["L2_is_weight_decay"]),
    use_averages: bool = True,
):
    return Optimizer(
        learn_rate,
        L2=L2,
        grad_clip=grad_clip,
        L2_is_weight_decay=L2_is_weight_decay,
        beta1=0.0,
        beta2=0.0,
        use_averages=use_averages,
    )


class Optimizer(object):
    """Do various flavours of stochastic gradient descent, with first and
    second order momentum. Currently support 'vanilla' SGD, Adam, and RAdam.
    """

    mom1: Dict[KeyT, FloatsXd]
    mom2: Dict[KeyT, FloatsXd]
    averages: Optional[Dict[KeyT, FloatsXd]]
    schedules: Dict[str, Generator]
    nr_update: Dict[KeyT, int]
    last_seen: Dict[KeyT, int]
    grad_clip: Schedule
    learn_rate: Schedule
    b1: Schedule
    b2: Schedule
    eps: Schedule
    L2: Schedule
    use_radam: bool
    L2_is_weight_decay: bool
    _radam_buffer: List[List[Optional[FloatsXd]]]
    _step: int
    _last_score: Optional[Tuple[int, float]]

    # This "locks" the class, so we get an error if you try to assign to
    # an unexpected variable.
    __slots__ = [
        "mom1",
        "mom2",
        "averages",
        "schedules",
        "nr_update",
        "last_seen",
        "grad_clip",
        "learn_rate",
        "b1",
        "b2",
        "eps",
        "L2",
        "use_radam",
        "L2_is_weight_decay",
        "_radam_buffer",
        "_step",
        "_last_score",
    ]

    def __init__(
        self,
        learn_rate: ScheduleT,
        *,
        L2: ScheduleT = ADAM_DEFAULTS["L2"],
        beta1: ScheduleT = ADAM_DEFAULTS["beta1"],
        beta2: ScheduleT = ADAM_DEFAULTS["beta2"],
        eps: ScheduleT = ADAM_DEFAULTS["eps"],
        grad_clip: ScheduleT = ADAM_DEFAULTS["grad_clip"],
        use_averages: bool = True,
        use_radam: bool = False,
        L2_is_weight_decay: bool = True,
    ):
        """
        Initialize an optimizer.

        learn_rate (float): The initial learning rate.
        L2 (float): The L2 regularization term.
        beta1 (float): First-order momentum.
        beta2 (float): Second-order momentum.
        eps (float): Epsilon term for Adam etc.
        grad_clip (float): Gradient clipping.
        use_averages (bool): Whether to track moving averages of the parameters.
        use_radam (bool): Whether to use the RAdam optimizer.
        L2_is_weight_decay (bool): Whether to interpret the L2 parameter as a
            weight decay term, in the style of the AdamW optimizer.
        """
        self._step = 0
        self._last_score = None
        self.mom1 = {}
        self.mom2 = {}
        if use_averages:
            self.averages = {}
        else:
            self.averages = None
        self.nr_update = defaultdict(int)
        self.last_seen = defaultdict(int)
        self._set_attr_or_schedule("grad_clip", grad_clip)
        self._set_attr_or_schedule("learn_rate", learn_rate)
        self._set_attr_or_schedule("b1", beta1)
        self._set_attr_or_schedule("b2", beta2)
        self._set_attr_or_schedule("eps", eps)
        self._set_attr_or_schedule("L2", L2)
        self.use_radam = use_radam
        self.L2_is_weight_decay = L2_is_weight_decay
        self._radam_buffer = [[None, None, None] for _ in range(10)]

    def _set_attr_or_schedule(self, name, value):
        if isinstance(value, (float, bool, int)):
            setattr(self, name, constant(value))
        elif isinstance(value, list):
            value = iter(value)
            setattr(self, name, _wrap_generator(name, value))
        elif isinstance(value, GeneratorType):
            setattr(self, name, _wrap_generator(name, value))
        elif isinstance(value, Schedule):
            setattr(self, name, value)
        else:
            err = f"Invalid schedule for '{name}' ({type(value)})"
            raise ValueError(err)

    def step_schedules(self):
        self._step += 1

    @property
    def last_score(self) -> Optional[Tuple[int, float]]:
        return self._last_score

    @last_score.setter
    def last_score(self, score: float):
        self._last_score = (self._step, score)

    @property
    def step(self) -> int:
        return self._step

    def _schedule_args(self, key: KeyT) -> Dict[str, Any]:
        return {
            "key": key,
            "last_score": self.last_score,
        }

    def __call__(
        self,
        key: Tuple[int, str],
        weights: FloatsXd,
        gradient: FloatsXd,
        *,
        lr_scale: float = 1.0,
    ):
        """Call the optimizer with weights and a gradient. The key is the
        identifier for the parameter, usually the node ID and parameter name.
        """
        if len(gradient) < 1:
            return weights, gradient

        ops = get_array_ops(weights)
        self.nr_update[key] += 1
        nr_upd = self.nr_update[key]
        schedule_args = self._schedule_args(key)

        if self.L2(self.step, **schedule_args) != 0 and not self.L2_is_weight_decay:
            gradient += self.L2(self.step, **schedule_args) * weights
        if self.grad_clip(self.step, **schedule_args):
            gradient = ops.clip_gradient(
                gradient,
                self.grad_clip(self.step, **schedule_args),
            )
        if self.use_radam:
            weights, gradient = self._radam(
                ops, weights, gradient, lr_scale, key, nr_upd
            )
        elif (
            self.b1(self.step, **schedule_args) > 0.0
            and self.b2(self.step, **schedule_args) > 0.0
        ):
            weights, gradient = self._adam(
                ops, weights, gradient, lr_scale, key, nr_upd
            )
        elif self.b2(self.step, **schedule_args) > 0.0:  # pragma: no cover
            raise NotImplementedError  # TODO: error message
        else:
            weights -= lr_scale * self.learn_rate(self.step, **schedule_args) * gradient
        gradient *= 0
        if self.L2(self.step, **schedule_args) != 0 and self.L2_is_weight_decay:
            weights -= (
                lr_scale
                * self.learn_rate(self.step, **schedule_args)
                * self.L2(self.step, **schedule_args)
                * weights
            )
        if self.averages is not None:
            if key not in self.averages:
                self.averages[key] = ops.alloc(weights.shape, dtype="float32")
            ops.update_averages(self.averages[key], weights, nr_upd)
        return weights, gradient

    def _radam(self, ops, weights, grad, lr_scale, key, nr_upd):
        if key not in self.mom1:
            self.mom1[key] = ops.alloc1f(weights.size)
        if key not in self.mom2:
            self.mom2[key] = ops.alloc1f(weights.size)

        weights_1D = ops.reshape1f(weights, weights.size)
        gradient_1D = ops.reshape1f(grad, grad.size)

        schedule_args = self._schedule_args(key)

        # While we port from the pytorch implementation, keep some of the same
        # naming
        state = {
            "step": self.nr_update[key],
            "exp_avg": self.mom1[key],
            "exp_avg_sq": self.mom2[key],
        }
        group = {
            "lr": self.learn_rate(self.step, **schedule_args),
            "betas": [
                self.b1(self.step, **schedule_args),
                self.b2(self.step, **schedule_args),
            ],
            "eps": self.eps(self.step, **schedule_args),
            "weight_decay": 0.0,
            "buffer": self._radam_buffer,
        }
        degenerated_to_sgd = True

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]

        # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        exp_avg_sq *= beta2
        exp_avg_sq += (1 - beta2) * (gradient_1D**2)
        # exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg *= beta1
        exp_avg += (1 - beta1) * gradient_1D

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
                weights_1D += -group["weight_decay"] * group["lr"] * weights_1D
            denom = ops.xp.sqrt(exp_avg_sq) + group["eps"]
            weights_1D += -step_size * group["lr"] * (exp_avg / denom)
        elif step_size > 0:
            if group["weight_decay"] != 0:
                weights_1D += -group["weight_decay"] * group["lr"] * weights_1D
            weights_1D += -step_size * group["lr"] * exp_avg
        return (
            ops.reshape_f(weights_1D, weights.shape),
            ops.reshape_f(gradient_1D, grad.shape),
        )

    def _adam(self, ops, weights, gradient, lr_scale, key, nr_upd):
        weights_1D = ops.reshape1f(weights, weights.size)
        gradient_1D = ops.reshape1f(gradient, gradient.size)

        schedule_args = self._schedule_args(key)

        if key not in self.mom1:
            self.mom1[key] = ops.alloc1f(weights.size)
        if key not in self.mom2:
            self.mom2[key] = ops.alloc1f(weights.size)
        mom1 = self.mom1[key]
        mom2 = self.mom2[key]
        b1 = self.b1(self.step, **schedule_args)
        b2 = self.b2(self.step, **schedule_args)
        fix1 = 1.0 - (b1**nr_upd)
        fix2 = 1.0 - (b2**nr_upd)
        lr = self.learn_rate(self.step, **schedule_args) * fix2**0.5 / fix1
        eps = self.eps(self.step, **schedule_args)
        # needs to be 1D going into the adam function
        weights_1D, gradient_1D, mom1, mom2 = ops.adam(
            weights_1D, gradient_1D, mom1, mom2, b1, b2, eps, lr * lr_scale
        )
        self.mom1[key] = mom1
        self.mom2[key] = mom2
        return (
            ops.reshape_f(weights_1D, weights.shape),
            ops.reshape_f(gradient_1D, gradient.shape),
        )


def _wrap_generator(attr_name: str, generator: Generator) -> Schedule[Any]:
    try:
        peek = next(generator)
    except (StopIteration, TypeError) as e:
        err = f"Invalid schedule for '{attr_name}' ({type(generator)})\n{e}"
        raise ValueError(err)
    return Schedule(
        "wrap_generator",
        _wrap_generator_schedule,
        attrs={
            "attr_name": attr_name,
            "last_step": -1,
            "last_value": peek,
            "generator": itertools.chain([peek], generator),
        },
    )


def _wrap_generator_schedule(schedule: Schedule, step, **kwargs) -> float:
    attr_name = schedule.attrs["attr_name"]
    last_step = schedule.attrs["last_step"]
    last_value = schedule.attrs["last_value"]
    generator = schedule.attrs["generator"]

    if step < last_step:
        raise ValueError(
            f"'step' of the generator-based schedule for {attr_name} must not decrease"
        )

    # Ensure that we have a value when we didn't step or when the
    # generator is exhausted.
    value = last_value

    for i in range(step - last_step):
        try:
            value = next(generator)
        except StopIteration:  # schedule exhausted, use last value
            break

    schedule.attrs["last_step"] = step
    schedule.attrs["last_value"] = value

    return value


__all__ = ["Adam", "RAdam", "SGD", "Optimizer", "ADAM_DEFAULTS", "SGD_DEFAULTS"]
