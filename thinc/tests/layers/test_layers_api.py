from thinc.api import registry
from thinc.types import Ragged, Padded
import numpy
import pytest


class FakeDoc:
    def to_array(self, attr_ids):
        return attr_ids


class FakeSpan:
    doc = FakeDoc()
    start = 0
    end = -1


array1d = numpy.asarray([1, 2, 3], dtype="f")
array2d = numpy.asarray([[1, 2, 3, 4], [4, 5, 3, 4]], dtype="f")
array2dint = numpy.asarray([[1, 2, 3], [4, 5, 6]], dtype="i")
array3d = numpy.zeros((3, 3, 3), dtype="f")
ragged = Ragged(array2d, numpy.asarray([1, 1], dtype="i"))
padded_data = (array3d, array1d, [1, 2, 3], [1, 2, 3])
padded = Padded(array3d, array1d, [1, 2, 3, 4], [1, 2, 3, 4])
doc = FakeDoc()
span = FakeSpan()


def assert_data_match(Y, out_data):
    assert type(Y) == type(out_data)
    if isinstance(out_data, numpy.ndarray):
        assert isinstance(Y, numpy.ndarray)
        assert out_data.ndim == Y.ndim
    elif isinstance(out_data, Ragged):
        assert isinstance(Y, Ragged)
        assert out_data.data.ndim == Y.data.ndim
        assert out_data.lengths.ndim == Y.lengths.ndim
    elif isinstance(out_data, (list, tuple)):
        assert isinstance(Y, (list, tuple))
        assert all(isinstance(x, numpy.ndarray) for x in Y)
    else:
        pytest.fail(f"wrong output of {type(Y)}: {Y}")


TEST_CASES = [
    # Array to array
    ("Dropout.v0", {}, array2d, array2d),
    ("Embed.v0", {}, array2dint, array2d),
    ("Embed.v0", {"nO": 4}, array2dint, array2d),
    ("ExtractWindow.v0", {}, array2d, array2d),
    ("HashEmbed.v0", {"nO": 1, "nV": 2}, array2d, array2d),
    ("LayerNorm.v0", {}, array2d, array2d),
    ("Linear.v0", {}, array2d, array2d),
    ("Maxout.v0", {}, array2d, array2d),
    ("Maxout.v0", {"normalize": True, "dropout": 0.2}, array2d, array2d),
    ("Maxout.v0", {"nO": 4, "nI": 4}, array2d, array2d),
    ("Mish.v0", {}, array2d, array2d),
    ("Mish.v0", {"nO": 4, "nI": 4}, array2d, array2d),
    ("Mish.v0", {"normalize": True, "dropout": 0.2}, array2d, array2d),
    ("MultiSoftmax.v0", {"nOs": (1, 3)}, array2d, array2d),
    ("ReLu.v0", {}, array2d, array2d),
    ("ReLu.v0", {"normalize": True, "dropout": 0.2}, array2d, array2d),
    ("Softmax.v0", {}, array2d, array2d),
    ("Softmax.v0", {"nO": 4, "nI": 4}, array2d, array2d),
    # Currently doesn't work because it requires spaCy:
    # ("StaticVectors.v0", array2d, array2d),
    # Ragged to array
    ("MaxPool.v0", {}, ragged, array2d),
    ("MeanPool.v0", {}, ragged, array2d),
    ("SumPool.v0", {}, ragged, array2d),
    # List to list
    ("BiLSTM.v0", {}, [array2d, array2d], [array2d, array2d]),
    ("LSTM.v0", {}, [array2d, array2d], [array2d, array2d]),
    # Other
    # fmt: off
    ("CauchySimilarity.v0", {}, (array2d, array2d), array1d),
    ("FeatureExtractor.v0", {"columns": [1, 2]}, [doc, doc, doc], [array2d, array2d, array2d]),
    ("FeatureExtractor.v0", {"columns": [1, 2]}, [span, span], [array2d, array2d]),
    ("ParametricAttention.v0", {}, ragged, ragged),
    ("SparseLinear.v0", {}, (numpy.asarray([1, 2, 3], dtype="uint64"), array1d, numpy.asarray([1, 1], dtype="i")), array2d),
    # fmt: on
]


@pytest.mark.parametrize("name,kwargs,in_data,out_data", TEST_CASES)
def test_layers_from_config(name, kwargs, in_data, out_data):
    cfg = {"@layers": name, **kwargs}
    filled = registry.fill_config({"config": cfg})
    model = registry.make_from_config(filled)["config"]
    model.initialize(in_data, out_data)
    Y, backprop = model(in_data)
    assert_data_match(Y, out_data)
    dX = backprop(Y)
    assert_data_match(dX, in_data)


@pytest.mark.parametrize("name", ["BiLSTM.v0", "LSTM.v0"])
@pytest.mark.parametrize(
    "data",
    [
        ragged,
        [array2d, array2d],
        ragged,
        # padded,
    ],
)
@pytest.mark.parametrize("transform", ["with_list.v0"])
def test_layers_list2list_transforms(name, transform, data):
    cfg = {"@layers": transform, "layer": {"@layers": name}}
    model = registry.make_from_config({"config": cfg})["config"]
    assert len(model.layers) == 1
    model.initialize(data, data)
    Y, backprop = model(data)
    assert_data_match(Y, data)
    dX = backprop(Y)
    assert_data_match(dX, data)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("Dropout.v0", {}),
        ("Embed.v0", {}),
        ("ExtractWindow.v0", {}),
        ("HashEmbed.v0", {"nO": 1, "nV": 2}),
        ("LayerNorm.v0", {}),
        ("Linear.v0", {}),
        ("Maxout.v0", {}),
        ("Mish.v0", {}),
        ("MultiSoftmax.v0", {"nOs": (1, 3)}),
        ("ReLu.v0", {}),
        ("Softmax.v0", {}),
    ],
)
@pytest.mark.parametrize(
    "transform",
    [
        "with_array.v0",
        # "with_reshape.v0"
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        array2d,
        ragged,
        [array2d, array2d],
        padded,
        # padded_data,
    ],
)
def test_layers_array2array_transforms(name, kwargs, transform, data):
    cfg = {"@layers": transform, "layer": {"@layers": name, **kwargs}}
    model = registry.make_from_config({"config": cfg})["config"]
    assert len(model.layers) == 1
    model.initialize(data, data)
    # Y, backprop = model(data)
    # assert_data_match(Y, out_data)
    # dX = backprop(Y)
    # assert_data_match(dX, in_data)
