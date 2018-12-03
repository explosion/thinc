# coding: utf8
"""Generators that provide different rates, schedules, decays or series."""

from __future__ import unicode_literals, division


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


def annealing(rate, decay, decay_steps, t=0.0):
    while True:
        if decay == 0.0:
            yield rate
        else:
            yield rate * decay ** (t / decay_steps)
            t += 1


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
