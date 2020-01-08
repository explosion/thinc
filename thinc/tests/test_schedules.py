from thinc.api import decaying, compounding, slanted_triangular


def test_decaying_rate():
    rates = decaying(0.001, 1e-4)
    rate = next(rates)
    assert rate == 0.001
    next_rate = next(rates)
    assert next_rate < rate
    assert next_rate > 0
    assert next_rate > next(rates)


def test_compounding_rate():
    rates = compounding(1, 16, 1.01)
    rate0 = next(rates)
    assert rate0 == 1.0
    rate1 = next(rates)
    rate2 = next(rates)
    rate3 = next(rates)
    assert rate3 > rate2 > rate1 > rate0
    assert (rate3 - rate2) > (rate2 - rate1) > (rate1 - rate0)


def test_slanted_triangular_rate():
    rates = slanted_triangular(1.0, 20.0, ratio=10)
    rate0 = next(rates)
    assert rate0 < 1.0
    rate1 = next(rates)
    assert rate1 > rate0
    rate2 = next(rates)
    assert rate2 < rate1
    rate3 = next(rates)
    assert rate0 < rate3 < rate2
