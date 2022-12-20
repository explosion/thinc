"""Generators that provide different rates, schedules, decays or series."""
from typing import Any, Callable, Dict, Generic, TypeVar
import numpy

from .config import registry

OutT = TypeVar("OutT")


class Schedule(Generic[OutT]):
    """Class for implementing Thinc schedules."""

    name: str
    _schedule: Callable
    _attrs: Dict[str, Any]

    __slots__ = ["name", "_schedule", "_attrs"]

    def __init__(
        self, name: str, schedule: Callable, *, attrs: Dict[str, Any] = {}
    ) -> None:
        """Initialize a new schedule.

        name (str): The name of the schedule type.
        schedule (Callable): The schedule function.
        """
        self.name = name
        self._schedule = schedule
        self._attrs = dict(attrs)

    def __call__(self, step: int, **extra) -> OutT:
        """Compute the schedule for a given step."""

        if step < 0:
            raise ValueError(f"Step must be non-negative, was: {step}")

        return self._schedule(self, step, **extra)

    @property
    def attrs(self):
        """Schedule attributes."""
        return self._attrs


@registry.schedules("constant_then.v1")
def constant_then(rate: OutT, steps: int, schedule: Schedule[OutT]) -> Schedule[OutT]:
    """Yield a constant rate for N steps, before starting a schedule."""
    return Schedule(
        "constant_then",
        _constant_then_schedule,
        attrs={"rate": rate, "steps": steps, "schedule": schedule},
    )


def _constant_then_schedule(schedule: Schedule, step: int, **kwargs) -> float:
    rate = schedule.attrs["rate"]
    steps = schedule.attrs["steps"]
    schedule = schedule.attrs["schedule"]

    if step < steps:
        return rate
    else:
        return schedule(step=step, **kwargs)


@registry.schedules("constant.v1")
def constant(rate: OutT) -> Schedule[OutT]:
    """Yield a constant rate."""
    return Schedule("constant", _constant_schedule, attrs={"rate": rate})


def _constant_schedule(schedule: Schedule, step: int, **kwargs) -> float:
    rate = schedule.attrs["rate"]
    return rate


@registry.schedules("decaying.v1")
def decaying(base_rate: float, decay: float, *, t: float = 0.0) -> Schedule[float]:
    """Yield an infinite series of linearly decaying values,
    following the schedule: base_rate * 1 / (1 + decay * (t + step))

    EXAMPLE:
        >>> learn_rates = decaying(0.001, 1e-4)
        >>> next(learn_rates)
        0.001
        >>> next(learn_rates)
        0.00999
    """
    return Schedule(
        "decaying",
        _decaying_schedule,
        attrs={"base_rate": base_rate, "decay": decay, "t": t},
    )


def _decaying_schedule(schedule: Schedule, step: int, **kwargs) -> float:
    base_rate = schedule.attrs["base_rate"]
    decay = schedule.attrs["decay"]
    t = schedule.attrs["t"]
    return base_rate * (1.0 / (1.0 + decay * (step + t)))


@registry.schedules("compounding.v1")
def compounding(
    start: float, stop: float, compound: float, *, t: float = 0.0
) -> Schedule[float]:
    """Yield an infinite series of compounding values. Each time the
    generator is called, a value is produced by multiplying the previous
    value by the compound rate.

    EXAMPLE:
        >>> sizes = compounding(1.0, 10.0, 1.5)
        >>> assert next(sizes) == 1.
        >>> assert next(sizes) == 1 * 1.5
        >>> assert next(sizes) == 1.5 * 1.5
    """
    return Schedule(
        "compounding",
        _compounding_schedule,
        attrs={"start": start, "stop": stop, "compound": compound, "t": t},
    )


def _compounding_schedule(schedule: Schedule, step: int, **kwargs) -> float:
    start = schedule.attrs["start"]
    stop = schedule.attrs["stop"]
    compound = schedule.attrs["compound"]
    t = schedule.attrs["t"]
    return _clip(start * (compound ** (step + t)), start, stop)


def _clip(value: float, start: float, stop: float) -> float:
    return max(value, stop) if (start > stop) else min(value, stop)


@registry.schedules("slanted_triangular.v1")
def slanted_triangular(
    max_rate: float,
    num_steps: int,
    *,
    cut_frac: float = 0.1,
    ratio: int = 32,
    t: float = 0.0,
) -> Schedule[float]:
    """Yield an infinite series of values according to Howard and Ruder's
    "slanted triangular learning rate" schedule.
    """
    cut = int(num_steps * cut_frac)
    return Schedule(
        "slanted_triangular",
        _slanted_triangular_schedule,
        attrs={
            "max_rate": max_rate,
            "cut": cut,
            "cut_frac": cut_frac,
            "ratio": ratio,
            "t": t,
        },
    )


def _slanted_triangular_schedule(schedule: Schedule, step: int, **kwargs) -> float:
    max_rate = schedule.attrs["max_rate"]
    cut = schedule.attrs["cut"]
    cut_frac = schedule.attrs["cut_frac"]
    ratio = schedule.attrs["ratio"]
    t = schedule.attrs["t"]

    t_step = step + t + 1.0
    if t_step < cut:
        p = t_step / cut
    else:
        p = 1 - ((t_step - cut) / (cut * (1 / cut_frac - 1)))
    return max_rate * (1 + p * (ratio - 1)) * (1 / ratio)


@registry.schedules("warmup_linear.v1")
def warmup_linear(
    initial_rate: float, warmup_steps: int, total_steps: int
) -> Schedule[float]:
    """Generate a series, starting from an initial rate, and then with a warmup
    period, and then a linear decline. Used for learning rates.
    """
    return Schedule(
        "warmup_linear",
        _warmup_linear_schedule,
        attrs={
            "initial_rate": initial_rate,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        },
    )


def _warmup_linear_schedule(schedule: Schedule, step: int, **kwargs) -> float:
    initial_rate = schedule.attrs["initial_rate"]
    warmup_steps = schedule.attrs["warmup_steps"]
    total_steps = schedule.attrs["total_steps"]

    if step < warmup_steps:
        factor = step / max(1, warmup_steps)
    else:
        factor = max(0.0, (total_steps - step) / max(1.0, total_steps - warmup_steps))
    return factor * initial_rate


@registry.schedules("cyclic_triangular.v1")
def cyclic_triangular(min_lr: float, max_lr: float, period: int) -> Schedule[float]:
    return Schedule(
        "cyclic_triangular",
        _cyclic_triangular_schedule,
        attrs={"min_lr": min_lr, "max_lr": max_lr, "period": period},
    )


def _cyclic_triangular_schedule(schedule: Schedule, step: int, **kwargs) -> float:
    min_lr = schedule.attrs["min_lr"]
    max_lr = schedule.attrs["max_lr"]
    period = schedule.attrs["period"]

    it = step + 1
    # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
    cycle = numpy.floor(1 + it / (2 * period))
    x = numpy.abs(it / period - 2 * cycle + 1)
    relative = max(0, 1 - x)
    return min_lr + (max_lr - min_lr) * relative


__all__ = [
    "cyclic_triangular",
    "warmup_linear",
    "constant",
    "constant_then",
    "decaying",
    "warmup_linear",
    "slanted_triangular",
    "compounding",
]
