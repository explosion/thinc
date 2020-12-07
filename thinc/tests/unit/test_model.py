# encoding: utf8
from __future__ import unicode_literals

import tempfile
import os
import pytest
import threading
import time

from thinc.neural._classes import model as base
from thinc.neural.ops import NumpyOps


@pytest.fixture
def model_with_no_args():
    model = base.Model()
    return model


def test_Model_defaults_to_name_model(model_with_no_args):
    assert model_with_no_args.name == "model"


def test_changing_instance_name_doesnt_change_class_name():
    model = base.Model()
    assert model.name != "changed"
    model.name = "changed"
    model2 = base.Model()
    assert model2.name != "changed"


def test_changing_class_name_doesnt_change_default_instance_name():
    model = base.Model()
    assert model.name != "changed"
    base.Model.name = "changed"
    assert model.name != "changed"
    # Reset state
    base.Model.name = "model"


def test_changing_class_name_doesnt_changes_nondefault_instance_name():
    model = base.Model(name="nondefault")
    assert model.name == "nondefault"
    base.Model.name = "changed"
    assert model.name == "nondefault"


def test_Model_defaults_to_cpu(model_with_no_args):
    assert isinstance(model_with_no_args.ops, NumpyOps)


def test_models_get_different_ids(model_with_no_args):
    model1 = base.Model()
    model2 = base.Model()
    assert model1.id != model2.id


def test_init_assigns_attributes():
    model = base.Model()
    model._mem
    assert model._layers == []


def test_init_installs_via_descriptions():
    def mock_install(attr, self):
        setattr(self, attr, "model=" + self.name)

    base.Model.descriptions = [("myattr", mock_install)]
    model = base.Model(name="model1")
    assert model.myattr == "model=%s" % "model1"
    model2 = base.Model(name="model2")
    assert model2.myattr == "model=%s" % "model2"


def test_init_calls_hooks():
    def mock_init_hook(self, *args, **kwargs):
        setattr(self, "hooked", (args, kwargs))

    base.Model.on_init_hooks = [mock_init_hook]
    model = base.Model(0, 1, 2)
    assert model.hooked == ((0, 1, 2), {})
    model2 = base.Model(value="something")
    assert model2.hooked == (tuple(), {"value": "something"})


def test_use_device():
    dev_id = id(base.Model.ops)
    with base.Model.use_device(base.Model.ops.device):
        assert id(base.Model.ops) == dev_id
    with base.Model.use_device("gpu"):
        assert id(base.Model.ops) != dev_id
    assert id(base.Model.ops) == dev_id


def test_bind_plus():
    with base.Model.define_operators({"+": lambda a, b: (a.name, b.name)}):
        m = base.Model(name="a") + base.Model(name="b")
        assert m == ("a", "b")


def test_plus_chain():
    with base.Model.define_operators({"+": lambda a, b: a}):
        m = (
            base.Model(name="a")
            + base.Model(name="b")
            + base.Model(name="c")
            + base.Model(name="d")
        )
        assert m.name == "a"


def test_overload_operators_in_subthread():
    """Test we can create a model in a child thread with overloaded operators."""
    # Worker1 will start and run, while worker 2 sleeps after Model.define_operators.
    # Without thread-safety, worker2 will find that its operator definitions
    # have been removed, causing an error.
    worker1 = threading.Thread(target=_overload_plus, args=("+", 0))
    worker2 = threading.Thread(target=_overload_plus, args=("*", 1,))
    worker2.start()
    worker1.start()
    worker1.join()
    worker2.join()

    worker1 = threading.Thread(target=_overload_plus, args=("+", 1))
    worker2 = threading.Thread(target=_overload_plus, args=("*", 0,))
    worker2.start()
    worker1.start()
    worker1.join()
    worker2.join()


def _overload_plus(operator, sleep):
    m1 = base.Model(name="a")
    m2 = base.Model(name="b")
    with base.Model.define_operators({operator: lambda a, b: a.name + b.name}):
        time.sleep(sleep)
        if operator == "+":
            value = m1 + m2
        else:
            value = m1 * m2
    assert value == "ab"
    assert base.Model._thread_local.operators == {}


def test_nested_operator_contexts():
    operator = "+"
    m1 = base.Model(name="a")
    m2 = base.Model(name="b")
    assert base.Model._thread_local.operators == {}
    with base.Model.define_operators({"+": lambda a, b: a.name + b.name}):
        value = m1 + m2
        with pytest.raises(TypeError):
            value = m1 * m2
        with base.Model.define_operators({"*": lambda a, b: a.name + b.name}):
            with pytest.raises(TypeError):
                value = m1 + m2
            value = m1 * m2
            with base.Model.define_operators({"-": lambda a, b: a.name + b.name}):
                with pytest.raises(TypeError):
                    value = m1 + m2
                value = m1 - m2
            with pytest.raises(TypeError):
                value = m1 + m2
            value = m1 * m2
        value = m1 + m2
        with pytest.raises(TypeError):
            value = m1 * m2
    assert value == "ab"
    assert base.Model._thread_local.operators == {}


@pytest.mark.parametrize("op", "+ - * @ / // % ** << >> & ^ |".split())
def test_all_operators(op):
    m1 = base.Model(name="a")
    m2 = base.Model(name="b")
    with base.Model.define_operators({op: lambda a, b: a.name + b.name}):
        if op == "+":
            value = m1 + m2
        else:
            with pytest.raises(TypeError):
                value = m1 + m2
        if op == "-":
            value = m1 - m2
        else:
            with pytest.raises(TypeError):
                value = m1 - m2

        if op == "*":
            value = m1 * m2
        else:
            with pytest.raises(TypeError):
                value = m1 * m2

        if op == "@":
            value = m1.__matmul__(m2)  # Be kind to Python 2...
        else:
            with pytest.raises(TypeError):
                value = m1.__matmul__(m2)

        if op == "/":
            value = m1 / m2
        else:
            with pytest.raises(TypeError):
                value = m1 / m2

        if op == "//":
            value = m1 // m2
        else:
            with pytest.raises(TypeError):
                value = m1 // m2
        if op == "^":
            value = m1 ^ m2
        else:
            with pytest.raises(TypeError):
                value = m1 ^ m2
        if op == "%":
            value = m1 % m2
        else:
            with pytest.raises(TypeError):
                value = m1 % m2
        if op == "**":
            value = m1 ** m2
        else:
            with pytest.raises(TypeError):
                value = m1 ** m2
        if op == "<<":
            value = m1 << m2
        else:
            with pytest.raises(TypeError):
                value = m1 << m2
        if op == ">>":
            value = m1 >> m2
        else:
            with pytest.raises(TypeError):
                value = m1 >> m2
        if op == "&":
            value = m1 & m2
        else:
            with pytest.raises(TypeError):
                value = m1 & m2
        if op == "^":
            value = m1 ^ m2
        else:
            with pytest.raises(TypeError):
                value = m1 ^ m2
        if op == "|":
            value = m1 | m2
        else:
            with pytest.raises(TypeError):
                value = m1 | m2  # noqa: F841
    assert base.Model._thread_local.operators == {}


def test_model_can_save_to_disk(model_with_no_args):
    temp_file = os.path.join(tempfile.mkdtemp(), "thinc_model")
    model_with_no_args.to_disk(temp_file)


def test_model_can_load_from_disk(model_with_no_args):
    temp_file = os.path.join(tempfile.mkdtemp(), "thinc_model")
    model_with_no_args.to_disk(temp_file)
    m2 = model_with_no_args.from_disk(temp_file)
    assert model_with_no_args.to_bytes() == m2.to_bytes()
