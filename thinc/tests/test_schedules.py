from thinc.api import decaying, compounding, slanted_triangular, constant_then
from thinc.api import constant, warmup_linear, cyclic_triangular
from thinc.optimizers import KeyT


STUB_KEY: KeyT = (0, "")


def test_decaying_rate():
    rates = decaying(0.001, 1e-4)
    rate = rates(step=0, key=STUB_KEY)
    assert rate == 0.001
    next_rate = rates(step=1, key=STUB_KEY)
    assert next_rate < rate
    assert next_rate > 0
    assert next_rate > rates(step=2, key=STUB_KEY)


def test_compounding_rate():
    rates = compounding(1, 16, 1.01)
    rate0 = rates(step=0, key=STUB_KEY)
    assert rate0 == 1.0
    rate1 = rates(step=1, key=STUB_KEY)
    rate2 = rates(step=2, key=STUB_KEY)
    rate3 = rates(step=3, key=STUB_KEY)
    assert rate3 > rate2 > rate1 > rate0
    assert (rate3 - rate2) > (rate2 - rate1) > (rate1 - rate0)


def test_slanted_triangular_rate():
    rates = slanted_triangular(1.0, 20.0, ratio=10)
    rate0 = rates(step=0, key=STUB_KEY)
    assert rate0 < 1.0
    rate1 = rates(step=1, key=STUB_KEY)
    assert rate1 > rate0
    rate2 = rates(step=2, key=STUB_KEY)
    assert rate2 < rate1
    rate3 = rates(step=3, key=STUB_KEY)
    assert rate0 < rate3 < rate2


def test_constant_then_schedule():
    rates = constant_then(1.0, 2, constant(100))
    assert rates(step=0, key=STUB_KEY) == 1.0
    assert rates(step=1, key=STUB_KEY) == 1.0
    assert rates(step=2, key=STUB_KEY) == 100
    assert rates(step=3, key=STUB_KEY) == 100


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
