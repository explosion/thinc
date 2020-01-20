from thinc.api import registry, with_padded, Dropout, get_current_ops
from thinc.types import Ragged, Padded
from thinc.util import has_torch
import numpy
import pytest


class FakeDoc:
    def to_array(self, attr_ids):
        return attr_ids


class FakeSpan:
    doc = FakeDoc()
    start = 0
    end = -1


OPS = get_current_ops()

array1d = OPS.xp.asarray([1, 2, 3], dtype="f")
array2d = OPS.xp.asarray([[1, 2, 3, 4], [4, 5, 3, 4]], dtype="f")
array2dint = OPS.xp.asarray([[1, 2, 3], [4, 5, 6]], dtype="i")
array3d = OPS.xp.zeros((3, 3, 3), dtype="f")
ragged = Ragged(array2d, OPS.xp.asarray([1, 1], dtype="i"))
padded = Padded(array3d, array1d, OPS.asarray([1, 2, 3, 4]), OPS.asarray([1, 2, 3, 4]))
doc = FakeDoc()
span = FakeSpan()
width = array2d.shape[1]


def assert_data_match(Y, out_data):
    assert type(Y) == type(out_data)
    if isinstance(out_data, OPS.xp.ndarray):
        assert isinstance(Y, OPS.xp.ndarray)
        assert out_data.ndim == Y.ndim
    elif isinstance(out_data, Ragged):
        assert isinstance(Y, Ragged)
        assert out_data.data.ndim == Y.data.ndim
        assert out_data.lengths.ndim == Y.lengths.ndim
    elif isinstance(out_data, Padded):
        assert isinstance(Y, Padded)
        assert out_data.data.ndim == Y.data.ndim
        assert out_data.size_at_t.ndim == Y.size_at_t.ndim
        assert len(out_data.lengths) == len(Y.lengths)
        assert len(out_data.indices) == len(Y.indices)
    elif isinstance(out_data, (list, tuple)):
        assert isinstance(Y, (list, tuple))
        assert all(isinstance(x, numpy.ndarray) for x in Y)
    else:
        pytest.fail(f"wrong output of {type(Y)}: {Y}")


TEST_CASES_SUMMABLE = [
    # Array to array
    ("Dropout.v0", {}, array2d, array2d),
    ("LayerNorm.v0", {}, array2d, array2d),
    ("Linear.v0", {}, array2d, array2d),
    ("Logistic.v0", {}, array2d, array2d),
    ("Maxout.v0", {}, array2d, array2d),
    ("Maxout.v0", {"normalize": True, "dropout": 0.2}, array2d, array2d),
    ("Maxout.v0", {"nO": 4, "nI": 4}, array2d, array2d),
    ("Mish.v0", {}, array2d, array2d),
    ("Mish.v0", {"nO": 4, "nI": 4}, array2d, array2d),
    ("Mish.v0", {"normalize": True, "dropout": 0.2}, array2d, array2d),
    ("ReLu.v0", {}, array2d, array2d),
    ("ReLu.v0", {"normalize": True, "dropout": 0.2}, array2d, array2d),
    ("Softmax.v0", {}, array2d, array2d),
    ("Softmax.v0", {"nO": 4, "nI": 4}, array2d, array2d),
    # fmt: off
    # List to list
    ("LSTM.v0", {"bi": False}, [array2d, array2d], [array2d, array2d]),
    pytest.param("PyTorchLSTM.v0", {"bi": False, "nO": width, "nI": width}, [array2d, array2d], [array2d, array2d], marks=pytest.mark.skipif(not has_torch, reason="needs PyTorch")),
    # fmt: on
]

TEST_CASES = [
    *TEST_CASES_SUMMABLE,
    pytest.param(
        "PyTorchLSTM.v0",
        {"bi": True, "nO": width * 2, "nI": width},
        [array2d, array2d],
        [array2d, array2d],
        marks=pytest.mark.skipif(not has_torch, reason="needs PyTorch"),
    ),
    ("LSTM.v0", {"bi": True}, [array2d, array2d], [array2d, array2d]),
    # Currently doesn't work because it requires spaCy:
    # ("StaticVectors.v0", array2d, array2d),
    # Ragged to array
    ("reduce_max.v0", {}, ragged, array2d),
    ("reduce_mean.v0", {}, ragged, array2d),
    ("reduce_sum.v0", {}, ragged, array2d),
    # fmt: off
    # Other
    ("expand_window.v0", {}, array2d, array2d),
    ("Embed.v0", {"nV": 1}, array2dint, array2d),
    ("Embed.v0", {"nO": 4, "nV": 1}, array2dint, array2d),
    ("HashEmbed.v0", {"nO": 1, "nV": 2}, array2d, array2d),
    ("MultiSoftmax.v0", {"nOs": (1, 3)}, array2d, array2d),
    ("CauchySimilarity.v0", {}, (array2d, array2d), array1d),
    ("FeatureExtractor.v0", {"columns": [1, 2]}, [doc, doc, doc], [array2d, array2d, array2d]),
    ("FeatureExtractor.v0", {"columns": [1, 2]}, [span, span], [array2d, array2d]),
    ("ParametricAttention.v0", {}, ragged, ragged),
    ("SparseLinear.v0", {}, (numpy.asarray([1, 2, 3], dtype="uint64"), array1d, numpy.asarray([1, 1], dtype="i")), array2d),
    ("remap_ids.v0", {"dtype": "f"}, ["a", 1, 5.0], array2d)
    # fmt: on
]


@pytest.mark.parametrize("name,kwargs,in_data,out_data", TEST_CASES)
def test_layers_from_config(name, kwargs, in_data, out_data):
    cfg = {"@layers": name, **kwargs}
    filled = registry.fill_config({"config": cfg})
    model = registry.make_from_config(filled)["config"]
    if "LSTM" in name:
        model = with_padded(model)
    model.initialize(in_data, out_data)
    Y, backprop = model(in_data, is_train=True)
    assert_data_match(Y, out_data)
    dX = backprop(Y)
    assert_data_match(dX, in_data)


@pytest.mark.parametrize("name,kwargs,in_data,out_data", TEST_CASES_SUMMABLE)
def test_layers_with_residual(name, kwargs, in_data, out_data):
    cfg = {"@layers": "residual.v0", "layer": {"@layers": name, **kwargs}}
    filled = registry.fill_config({"config": cfg})
    model = registry.make_from_config(filled)["config"]
    if "LSTM" in name:
        model = with_padded(model)
    model.initialize(in_data, out_data)
    Y, backprop = model(in_data, is_train=True)
    assert_data_match(Y, out_data)
    dX = backprop(Y)
    assert_data_match(dX, in_data)


@pytest.mark.parametrize("data", [array2d, ragged, padded, [array2d, array2d]])
def test_dropout(data):
    model = Dropout(0.2)
    model.initialize(data, data)
    Y, backprop = model(data, is_train=False)
    assert_data_match(Y, data)
    dX = backprop(Y)
    assert_data_match(dX, data)
