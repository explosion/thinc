import pytest
import srsly
from thinc.api import with_array, Linear, Maxout, chain, Model
from thinc.api import serialize_attr, deserialize_attr


@pytest.fixture
def linear():
    return Linear(5, 3)


class SerializableAttr:
    value = "foo"

    def to_bytes(self):
        return self.value.encode("utf8")

    def from_bytes(self, data):
        self.value = f"{data.decode('utf8')} from bytes"
        return self


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


def test_simple_model_roundtrip_bytes_serializable_attrs():
    attr = SerializableAttr()
    assert attr.value == "foo"
    assert attr.to_bytes() == b"foo"
    model = Model("test", lambda X: (X, lambda dY: dY), attrs={"test": attr})
    with pytest.raises(TypeError):
        # SerializableAttr can't be serialized with msgpack
        model.to_bytes()

    @serialize_attr.register(SerializableAttr)
    def serialize_attr_custom(_, value, name, model):
        return value.to_bytes()

    @deserialize_attr.register(SerializableAttr)
    def deserialize_attr_custom(_, value, name, model):
        return SerializableAttr().from_bytes(value)

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


def test_serialize_attrs():
    fwd = lambda X: (X, lambda dY: dY)

    attrs = {"test": "foo"}
    model1 = Model("test", fwd, attrs=attrs)
    bytes_attr = serialize_attr(model1.get_attr("test"), attrs["test"], "test", model1)
    assert bytes_attr == srsly.msgpack_dumps("foo")
    model2 = Model("test", fwd, attrs={"test": ""})
    result = deserialize_attr(model2.get_attr("test"), bytes_attr, "test", model2)
    assert result == "foo"

    # Test objects with custom serialization functions
    @serialize_attr.register(SerializableAttr)
    def serialize_attr_custom(_, value, name, model):
        return value.to_bytes()

    @deserialize_attr.register(SerializableAttr)
    def deserialize_attr_custom(_, value, name, model):
        return SerializableAttr().from_bytes(value)

    attrs = {"test": SerializableAttr()}
    model3 = Model("test", fwd, attrs=attrs)
    bytes_attr = serialize_attr(model3.get_attr("test"), attrs["test"], "test", model3)
    assert bytes_attr == b"foo"
    model4 = Model("test", fwd, attrs=attrs)
    assert model4.get_attr("test").value == "foo"
    result = deserialize_attr(model4.get_attr("test"), bytes_attr, "test", model4)
    assert result.value == "foo from bytes"
