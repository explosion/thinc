# coding: utf8
"""Generators that provide different rates, schedules, decays or series."""
from __future__ import unicode_literals, division
import numpy
from ._registry import registry


@registry.schedules.register("constant_then.v1")
def constant_then(rate, steps, schedule):
    """Yield a constant rate for N steps, before starting a schedule."""
    for i in range(steps):
        yield rate
    for value in schedule:
        yield value


@registry.schedules.register("constant.v1")
def constant(rate):
    while True:
        yield rate


@registry.schedules.register("decaying.v1")
def decaying(base_rate, decay, t=0):
    """Yield an infinite series of linearly decaying values,
    following the schedule:

        base_rate * 1/(1+decay*t)

    Example:

        >>> learn_rates = linear_decay(0.001, 1e-4)
        >>> next(learn_rates)
        0.001
        >>> next(learn_rates)
        0.00999
    """
    while True:
        yield base_rate * (1.0 / (1.0 + decay * t))
        t += 1


@registry.schedules.register("compounding.v1")
def compounding(start, stop, compound, t=0.0):
    """Yield an infinite series of compounding values. Each time the
    generator is called, a value is produced by multiplying the previous
    value by the compound rate.

    EXAMPLE:
      >>> sizes = compounding(1., 10., 1.5)
      >>> assert next(sizes) == 1.
      >>> assert next(sizes) == 1 * 1.5
      >>> assert next(sizes) == 1.5 * 1.5
    """
    curr = float(start)
    while True:
        yield _clip(curr, start, stop)
        curr *= compound


def _clip(value, start, stop):
    return max(value, stop) if (start > stop) else min(value, stop)



@registry.schedules.register("slanted_triangular.v1")
def slanted_triangular(max_rate, num_steps, cut_frac=0.1, ratio=32, decay=1, t=0.0):
    """Yield an infinite series of values according to Howard and Ruder's
    "slanted triangular learning rate" schedule.
    """
    cut = int(num_steps * cut_frac)
    while True:
        t += 1
        if t < cut:
            p = t / cut
        else:
            p = 1 - ((t - cut) / (cut * (1 / cut_frac - 1)))
        learn_rate = max_rate * (1 + p * (ratio - 1)) * (1 / ratio)
        yield learn_rate


@registry.schedules.register("warmup_linear.v1")
def warmup_linear(initial_rate, warmup_steps, total_steps):
    """Generate a series, starting from an initial rate, and then with a warmup
    period, and then a linear decline. Used for learning rates.
    """
    step = 0
    while True:
        if step < warmup_steps:
            factor = step / max(1, warmup_steps)
        else:
            factor = max(
                0.0, (total_steps - step) / max(1.0, total_steps - warmup_steps)
            )
        yield factor * initial_rate
        step += 1


@registry.schedules.register("cyclic_triangular.v1")
def cyclic_triangular(min_lr, max_lr, period):
    it = 1
    while True:
        # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
        cycle = numpy.floor(1 + it / (2 * period))
        x = numpy.abs(it / period - 2 * cycle + 1)
        relative = max(0, 1 - x)
        yield min_lr + (max_lr - min_lr) * relative
        it += 1


# Deprecated

def annealing(rate, decay, decay_steps, t=0.0):
    while True:
        if decay == 0.0:
            yield rate
        else:
            yield rate * decay ** (t / decay_steps)
            t += 1


def annealing_cos(start, end, step=0.001):
    pct = step
    while True:
        cos_out = numpy.cos(numpy.pi * pct) + 1
        yield end + (start-end)/2 * cos_out
        pct += step


