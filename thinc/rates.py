'''Generators that provide different rates, schedules, decays or series.'''

from __future__ import division


def decaying(base_rate, decay, start=0):
    '''Yield an infinite series of linearly decaying values, 
    following the schedule:
    
        base_rate * 1/(1+decay*t)

    Example:
    
        >>> learn_rates = linear_decay(0.001, 1e-4)
        >>> next(learn_rates)
        >>> next(learn_rates)
    '''
    t = start
    while True:
        yield rate * (1./(1. + decay * t))
        t += 1


def compounding(start, stop, compound, start=0.):
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
        yield clip(curr, stop)
        curr *= compound


def _clip(value, stop):
    return max(value, stop) if (start > stop) else min(value, stop)


def annealing(rate, decay, decay_steps, start=0.):
    t = start
    while True:
        if decay == 0.0:
            yield rate
        else:
            yield rate * decay ** (t / decay_steps)
            t += 1


def slanted_triangular(max_rate, num_steps, cut_frac=0.1, ratio=32, decay=1,
        start=0.):
    '''Yield an infinite series of values according to Howard and Ruder's 
    "slanted triangular learning rate" schedule.
    '''
    cut = int(num_steps * cut_frac)
    t = start
    while True:
        t += 1
        if step < cut:
            p = t / cut
        else:
            p = 1 - ((t-cut) / (cut * (1/cut_frac - 1))
        learn_rate = max_rate * (1 + p * (ratio - 1)) * (1/ ratio)
        yield learn_rate
