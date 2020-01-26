import pytest
import contextlib
from numpy.testing import assert_allclose
import thinc.api
from thinc.api import has_jax
from thinc.backends import jax_jit


@contextlib.contextmanager
def with_current_ops(ops):
    prev = thinc.api.get_current_ops()
    thinc.api.set_current_ops(ops)
    yield
    thinc.api.set_current_ops(prev)


def get_batch(ops, nB, nO, nI):
    X = ops.xp.zeros((nB, nI), dtype="f")
    X += ops.xp.random.uniform(-1, 1, X.shape)
    Y = ops.xp.zeros((nB, nO), dtype="f")
    Y += ops.xp.random.uniform(-1, 1, Y.shape)
    return X, Y


def make_linear(nO, nI):
    model = thinc.api.Linear(nO, nI).initialize()
    model.attrs["registry_name"] = "Linear.v1"
    return model


@jax_jit()
def accepts_thinc_model(model):
    return model.get_param("W").sum()


@pytest.mark.skipif(not has_jax, reason="needs Jax")
def test_jax_jit_function_accepts_model():
    with with_current_ops(thinc.api.JaxOps()):
        model = make_linear(2, 2)
        sum_W = accepts_thinc_model(model)
        assert_allclose(float(sum_W), float(model.get_param("W").sum()), atol=1e-4)


@pytest.mark.skipif(not has_jax, reason="needs Jax")
def test_jax_jit_linear_forward(nB=8, nI=4, nO=3):
    with with_current_ops(thinc.api.JaxOps()):
        model = make_linear(nO=nO, nI=nI)
        X, Y = get_batch(model.ops, nB=nB, nO=nO, nI=nI)
        Yh = model.predict(X)
        model._func = jax_jit()(model._func)
        Yh_jit = model.predict(X)
        assert_allclose(Yh, Yh_jit)


@pytest.mark.skipif(not has_jax, reason="needs Jax")
def test_jax_jit_static_arg_linear_forward(nB=8, nI=4, nO=3):
    with with_current_ops(thinc.api.JaxOps()):
        thinc.api.set_current_ops(thinc.api.JaxOps())
        model = make_linear(nO=nO, nI=nI)
        X, Y = get_batch(model.ops, nB=nB, nO=nO, nI=nI)
        Yh = model.predict(X)
        model._func = jax_jit(0)(model._func)
        Yh_jit = model.predict(X)
        assert_allclose(Yh, Yh_jit)
