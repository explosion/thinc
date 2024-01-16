from itertools import islice

import pytest

from thinc.api import (
    compounding,
    constant,
    constant_then,
    cyclic_triangular,
    decaying,
    slanted_triangular,
    warmup_linear,
)
from thinc.schedules import plateau


def test_decaying_rate():
    rates = decaying(0.001, 1e-4)
    rate = rates(step=0)
    assert rate == 0.001
    next_rate = rates(step=1)
    assert next_rate < rate
    assert next_rate > 0
    assert next_rate > rates(step=2)

    rates_offset = decaying(0.001, 1e-4, t=1.0)
    assert rates(step=1) == rates_offset(step=0)
    assert rates(step=2) == rates_offset(step=1)


def test_compounding_rate():
    rates = compounding(1, 16, 1.01)
    rate0 = rates(step=0)
    assert rate0 == 1.0
    rate1 = rates(step=1)
    rate2 = rates(step=2)
    rate3 = rates(step=3)
    assert rate3 > rate2 > rate1 > rate0
    assert (rate3 - rate2) > (rate2 - rate1) > (rate1 - rate0)

    rates_offset = compounding(1, 16, 1.01, t=1.0)
    assert rates(step=1) == rates_offset(step=0)
    assert rates(step=2) == rates_offset(step=1)


def test_slanted_triangular_rate():
    rates = slanted_triangular(1.0, 20.0, ratio=10)
    rate0 = rates(step=0)
    assert rate0 < 1.0
    rate1 = rates(step=1)
    assert rate1 > rate0
    rate2 = rates(step=2)
    assert rate2 < rate1
    rate3 = rates(step=3)
    assert rate0 < rate3 < rate2

    rates_offset = slanted_triangular(1.0, 20.0, ratio=10, t=1.0)
    assert rates(step=1) == rates_offset(step=0)
    assert rates(step=2) == rates_offset(step=1)


def test_constant_then_schedule():
    rates = constant_then(1.0, 2, constant(100))
    assert rates(step=0) == 1.0
    assert rates(step=1) == 1.0
    assert rates(step=2) == 100
    assert rates(step=3) == 100


def test_constant():
    rates = constant(123)
    assert rates(step=0, key=(0, "")) == 123
    assert rates(step=0, key=(0, "")) == 123


def test_warmup_linear():
    rates = warmup_linear(1.0, 2, 10)
    expected = [0.0, 0.5, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
    for i in range(11):
        assert rates(step=i, key=(0, "")) == expected[i]


def test_cyclic_triangular():
    rates = cyclic_triangular(0.1, 1.0, 2)
    expected = [0.55, 1.0, 0.55, 0.1, 0.55, 1.0, 0.55, 0.1, 0.55, 1.0]
    for i in range(10):
        assert rates(step=i, key=(0, "")) == expected[i]


def test_plateau():
    schedule = plateau(2, 0.5, constant(1.0))
    assert schedule(step=0, last_score=None) == 1.0
    assert schedule(step=1, last_score=(1, 1.0)) == 1.0  # patience == 0
    assert schedule(step=2, last_score=(2, 1.0)) == 1.0  # patience == 1
    assert schedule(step=3, last_score=None) == 1.0  # patience == 1
    assert schedule(step=4, last_score=(4, 1.0)) == 0.5  # patience == 2, reset
    assert schedule(step=5, last_score=(4, 1.0)) == 0.5  # patience == 0
    assert schedule(step=6, last_score=(6, 0.9)) == 0.5  # patience == 1
    assert schedule(step=7, last_score=(7, 2.0)) == 0.5  # patience == 0
    assert schedule(step=8, last_score=(8, 1.0)) == 0.5  # patience == 1
    assert schedule(step=9, last_score=(9, 2.0)) == 0.25  # patience == 2, reset

    with pytest.raises(ValueError, match=r"Expected score with step"):
        schedule(step=1, last_score=(1, 1.0)) == 1.0


def test_to_generator():
    rates = warmup_linear(1.0, 2, 10)
    expected = [0.0, 0.5, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
    assert list(islice(rates.to_generator(), len(expected))) == expected
