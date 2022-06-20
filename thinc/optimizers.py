import math

from typing import Dict, Optional, Union, Tuple, List, cast
from typing import Generic, Any, TypeVar, Callable
from collections import defaultdict

from .backends import get_array_ops
from .types import Generator, FloatsXd
from .config import registry


KeyT = Tuple[int, str]
FloatOrSeq = Union[float, List[float], Generator]
IntOrSeq = Union[int, List[int], Generator]
GradT = TypeVar("GradT")
WeightT = TypeVar("WeightT")


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
    learn_rate: FloatOrSeq = ADAM_DEFAULTS["learn_rate"],
    *,
    beta1: FloatOrSeq = ADAM_DEFAULTS["beta1"],
    beta2: FloatOrSeq = ADAM_DEFAULTS["beta2"],
    eps: FloatOrSeq = ADAM_DEFAULTS["eps"],
    L2: FloatOrSeq = ADAM_DEFAULTS["L2"],
    L2_is_weight_decay: bool = cast(bool, ADAM_DEFAULTS["L2_is_weight_decay"]),
    grad_clip: FloatOrSeq = ADAM_DEFAULTS["grad_clip"],
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
    learn_rate: FloatOrSeq = ADAM_DEFAULTS["learn_rate"],
    *,
    L2: FloatOrSeq = ADAM_DEFAULTS["L2"],
    beta1: FloatOrSeq = ADAM_DEFAULTS["beta1"],
    beta2: FloatOrSeq = ADAM_DEFAULTS["beta2"],
    eps: FloatOrSeq = ADAM_DEFAULTS["eps"],
    grad_clip: FloatOrSeq = ADAM_DEFAULTS["grad_clip"],
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
    learn_rate: FloatOrSeq,
    *,
    L2: FloatOrSeq = SGD_DEFAULTS["L2"],
    grad_clip: FloatOrSeq = SGD_DEFAULTS["grad_clip"],
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


class OptimizerABC(Generic[GradT, WeightT]):
    def __init__(self, **kwargs: Any) -> None:
        ...

    def __call__(
        self,
        key: Tuple[int, str],
        weights: WeightT,
        gradient: GradT,
        **kwargs: Any
    ) -> Tuple[WeightT, GradT]:
        ...


class Optimizer(OptimizerABC):
    """Do various flavours of stochastic gradient descent, with first and
    second order momentum. Currently support 'vanilla' SGD, Adam, and RAdam.
    """

    mom1: Dict[KeyT, FloatsXd]
    mom2: Dict[KeyT, FloatsXd]
    averages: Optional[Dict[KeyT, FloatsXd]]
    schedules: Dict[str, Generator]
    nr_update: Dict[KeyT, int]
    last_seen: Dict[KeyT, int]
    learn_rate: float
    grad_clip: float
    b1: float
    b2: float
    eps: float
    L2: float
    use_radam: bool
    L2_is_weight_decay: bool
    _radam_buffer: List[List[Optional[FloatsXd]]]

    # This "locks" the class, so we get an error if you try to assign to
    # an unexpected variable.
    __slots__ = [
        "mom1",
        "mom2",
        "averages",
        "learn_rate",
        "schedules",
        "nr_update",
        "last_seen",
        "grad_clip",
        "b1",
        "b2",
        "eps",
        "L2",
        "use_radam",
        "L2_is_weight_decay",
        "_radam_buffer",
    ]

    def __init__(
        self,
        learn_rate: FloatOrSeq,
        *,
        L2: FloatOrSeq = ADAM_DEFAULTS["L2"],
        beta1: FloatOrSeq = ADAM_DEFAULTS["beta1"],
        beta2: FloatOrSeq = ADAM_DEFAULTS["beta2"],
        eps: FloatOrSeq = ADAM_DEFAULTS["eps"],
        grad_clip: FloatOrSeq = ADAM_DEFAULTS["grad_clip"],
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
        self.mom1 = {}
        self.mom2 = {}
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
            try:
                setattr(self, name, next(value))
            except (StopIteration, TypeError) as e:
                err = f"Invalid schedule for '{name}' ({type(value)})\n{e}"
                raise ValueError(err)

    def step_schedules(self):
        for key, schedule in self.schedules.items():
            try:
                value = next(schedule)
            except StopIteration:  # schedule exhausted, use last value
                value = getattr(self, key)
            setattr(self, key, value)

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
        if self.L2 != 0 and not self.L2_is_weight_decay:
            gradient += self.L2 * weights
        if self.grad_clip:
            gradient = ops.clip_gradient(gradient, self.grad_clip)
        if self.use_radam:
            weights, gradient = self._radam(
                ops, weights, gradient, lr_scale, key, nr_upd
            )
        elif self.b1 > 0.0 and self.b2 > 0.0:
            weights, gradient = self._adam(
                ops, weights, gradient, lr_scale, key, nr_upd
            )
        elif self.b2 > 0.0:  # pragma: no cover
            raise NotImplementedError  # TODO: error message
        else:
            weights -= lr_scale * self.learn_rate * gradient
        gradient *= 0
        if self.L2 != 0 and self.L2_is_weight_decay:
            weights -= lr_scale * self.learn_rate * self.L2 * weights
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

        # While we port from the pytorch implementation, keep some of the same
        # naming
        state = {
            "step": self.nr_update[key],
            "exp_avg": self.mom1[key],
            "exp_avg_sq": self.mom2[key],
        }
        group = {
            "lr": self.learn_rate,
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
        if key not in self.mom1:
            self.mom1[key] = ops.alloc1f(weights.size)
        if key not in self.mom2:
            self.mom2[key] = ops.alloc1f(weights.size)
        mom1 = self.mom1[key]
        mom2 = self.mom2[key]
        b1 = self.b1
        b2 = self.b2
        fix1 = 1.0 - (b1**nr_upd)
        fix2 = 1.0 - (b2**nr_upd)
        lr = self.learn_rate * fix2**0.5 / fix1
        eps = self.eps
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


class SWA(OptimizerABC):
    """
    After the optimization ran 'start_step' number of steps
    it switches the learning-rate of the optimizer and starts
    recording the moving averages of all weights.
    """
    optimizer: Optimizer
    start_step: int
    freq: int
    learn_rate: FloatOrSeq
    nr_update: Dict[Tuple[int, str], int]
    run_swa: bool
    averages: Dict[Tuple[int, str], FloatsXd]

    def __init__(
        self,
        optimizer: Optimizer,
        learn_rate: FloatOrSeq,
        *,
        freq: int = 1,
        start_step: int = 0
    ):
        """
        optimizer: The inner optimizer whose iterates to average.
        learn_rate: Learning-rate (or schedule) to run with after
            the averaging is turned on.
        freq: Every 'freq' step the moving-averages will be updated.
        start_step: Start the averaging after start_step number of updates.
        """
        self.optimizer = optimizer
        self.start_step = start_step
        self.freq = freq
        self.learn_rate = learn_rate
        self.nr_update = defaultdict(int)
        self.run_swa = False
        self.averages = {}

    def _update_averages(
        self,
        key: Tuple[int, str],
        weights: FloatsXd,
    ):
        """
        Update moving average of weights.
        """
        if key not in self.averages:
            ops = get_array_ops(weights)
            self.averages[key] = ops.alloc(weights.shape, dtype="float32")
            self.averages[key] += weights
        else:
            self.averages[key] *= self.nr_update[key]
            self.averages[key] += weights
            self.averages[key] /= self.nr_update[key] + 1

    def _swap_lr(self):
        """
        Overwrite the optimizer's learning rate.
        """
        self.optimizer._set_attr_or_schedule("learn_rate", self.lr)

    def __call__(
        self,
        key: Tuple[int, str],
        weights: FloatsXd,
        gradient: FloatsXd,
        *,
        lr_scale: float = 1.0,
    ):
        self.nr_update[key] += 1
        nr_update = self.nr_update[key]
        weights, gradient = self.optimizer(key, weights, gradient, lr_scale=lr_scale)
        # Update running mean
        if nr_update == self.start_step:
            self._swap_lr()
        if nr_update >= self.start_step and nr_update % self.freq == 0:
            self._update_averages(key, weights)

        return weights, gradient


class Lookahead(OptimizerABC):
    """
    Keeps a "slow" exponential moving average
    of all "fast" weights. At each 'freq' update
    it updates the slow-weights and replaces the
    "fast" weights with them.
    """
    optimizer: Optimizer
    nr_update: Dict[Tuple[int, str], int]
    freq: int
    pullback: float
    averages: Dict[Tuple[int, str], FloatsXd]

    def __init__(self, optimizer: Optimizer, *, freq: int, pullback: float):
        """
        optimizer: The inner optimizer whose iterates to average.
        freq: Every 'freq' updates synchronize fast and slow weights and update
            moving-average.
        pullback: trades off the contribution of the new weight iterate
            and the moving average.
        """
        self.optimizer = optimizer
        self.nr_update: Dict[Tuple[int, str], int] = defaultdict(int)
        self.freq = freq
        self.pullback = pullback
        self.averages: Dict[Tuple[int, str], FloatsXd] = {}

    def _update_averages(
        self,
        key: Tuple[int, str],
        weights: FloatsXd,
    ):
        """
        Update exponental moving average of weights.
        """
        if key not in self.averages:
            ops = get_array_ops(weights)
            self.averages[key] = ops.alloc(weights.shape, dtype="float32")
            self.averages[key] += weights
        else:
            average = self.averages[key]
            self.averages[key] += self.pullback * (weights - average)

    def __call__(
        self,
        key: Tuple[int, str],
        weights: FloatsXd,
        gradient: FloatsXd,
        *,
        lr_scale: float = 1.0,
    ):
        self.nr_update[key] += 1
        nr_update = self.nr_update[key]
        weights, gradient = self.optimizer(key, weights, gradient, lr_scale=lr_scale)
        # Update running mean and return interpolated weights
        if nr_update % self.freq == 0:
            self._update_averages(key, weights)
            weights = self.averages[key]
        return weights, gradient


@registry.optimizers("SWA.v1")
def build_SWA(
    optimizer_factory: Callable[[], Optimizer],
    learn_rate: float,
    start_step: int,
    freq: int = 5
) -> SWA:
    optimizer = optimizer_factory()
    return SWA(
        optimizer=optimizer,
        learn_rate=learn_rate,
        freq=freq,
        start_step=start_step
    )


@registry.optimizers("Lookahead.v1")
def build_Lookahead(
    optimizer_factory: Callable[[], Optimizer],
    freq: int = 6,
    pullback: float = 0.5
) -> Lookahead:
    optimizer = optimizer_factory()
    return Lookahead(
        optimizer=optimizer,
        freq=freq,
        pullback=pullback,
    )


__all__ = [
    "Adam",
    "RAdam",
    "SGD",
    "Optimizer",
    "SWA",
    "Lookahead",
    "ADAM_DEFAULTS",
    "SGD_DEFAULTS"
]
