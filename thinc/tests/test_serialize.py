import pytest
import srsly
from thinc.api import with_array, Linear, Maxout, chain, Model
from thinc.api import serialize_attr, deserialize_attr


@pytest.fixture
def linear():
    return Linear(5, 3)


@pytest.fixture
def serializable_attr():
    class SerializableAttr:
        value = "foo"

        def to_bytes(self):
            return self.value.encode("utf8")

        def from_bytes(self, data):
            self.value = f"{data.decode('utf8')} from bytes"

    return SerializableAttr()


def test_pickle_with_flatten(linear):
    Xs = [linear.ops.alloc_f2d(2, 3), linear.ops.alloc_f2d(4, 3)]
    model = with_array(linear)
    pickled = srsly.pickle_dumps(model)
    loaded = srsly.pickle_loads(pickled)
    Ys = loaded.predict(Xs)
    assert len(Ys) == 2
    assert Ys[0].shape == (Xs[0].shape[0], linear.get_dim("nO"))
    assert Ys[1].shape == (Xs[1].shape[0], linear.get_dim("nO"))


def test_simple_model_roundtrip_bytes():
    model = Maxout(5, 10, nP=2)
    b = model.get_param("b")
    b += 1
    data = model.to_bytes()
    b = model.get_param("b")
    b -= 1
    model = model.from_bytes(data)
    assert model.get_param("b")[0, 0] == 1


def test_simple_model_roundtrip_bytes_serializable_attrs(serializable_attr):
    attr = serializable_attr
    assert attr.value == "foo"
    assert attr.to_bytes() == b"foo"
    model = Model("test", lambda X: (X, lambda dY: dY), attrs={"test": attr})
    model_bytes = model.to_bytes()
    model = model.from_bytes(model_bytes)
    assert model.has_attr("test")
    assert model.get_attr("test").value == "foo from bytes"


def test_multi_model_roundtrip_bytes():
    model = chain(Maxout(5, 10, nP=2), Maxout(2, 3))
    b = model.layers[0].get_param("b")
    b += 1
    b = model.layers[1].get_param("b")
    b += 2
    data = model.to_bytes()
    b = model.layers[0].get_param("b")
    b -= 1
    b = model.layers[1].get_param("b")
    b -= 2
    model = model.from_bytes(data)
    assert model.layers[0].get_param("b")[0, 0] == 1
    assert model.layers[1].get_param("b")[0, 0] == 2


def test_multi_model_load_missing_dims():
    model = chain(Maxout(5, 10, nP=2), Maxout(2, 3))
    b = model.layers[0].get_param("b")
    b += 1
    b = model.layers[1].get_param("b")
    b += 2
    data = model.to_bytes()

    model2 = chain(Maxout(5, nP=None), Maxout(nP=None))
    model2 = model2.from_bytes(data)
    assert model2.layers[0].get_param("b")[0, 0] == 1
    assert model2.layers[1].get_param("b")[0, 0] == 2


def test_serialize_attrs(serializable_attr):
    fwd = lambda X: (X, lambda dY: dY)
    # Test msgpack-serializable attrs
    model1 = Model("test", fwd, attrs={"test": "foo"})
    bytes_attr = serialize_attr("foo", "test", model1)
    assert bytes_attr == srsly.msgpack_dumps("foo")
    model2 = Model("test", fwd, attrs={"test": ""})
    deserialize_attr(bytes_attr, "test", model2)
    assert model2.get_attr("test") == "foo"
    # Test objects with to_bytes/from_bytes
    model3 = Model("test", fwd, attrs={"test": serializable_attr})
    bytes_attr = serialize_attr(serializable_attr, "test", model3)
    assert bytes_attr == b"foo"
    model4 = Model("test", fwd, attrs={"test": serializable_attr})
    deserialize_attr(bytes_attr, "test", model4)
    model4.get_attr("test").value == "foo from bytes"
