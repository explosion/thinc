from thinc.api import registry, with_padded
from thinc.types import Ragged
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
ragged = Ragged(array2d, numpy.asarray([1, 1], dtype="i"))
doc = FakeDoc()
span = FakeSpan()
width = array2d.shape[1]


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
    ("LSTM.v0", {"bi": False}, [array2d, array2d], [array2d, array2d]),
    ("LSTM.v0", {"bi": True}, [array2d, array2d], [array2d, array2d]),
    (
        "PyTorchLSTM.v0",
        {"bi": False, "nO": width, "nI": width},
        [array2d, array2d],
        [array2d, array2d],
    ),
    (
        "PyTorchLSTM.v0",
        {"bi": True, "nO": width * 2, "nI": width},
        [array2d, array2d],
        [array2d, array2d],
    ),
    # ("PyTorchBiLSTM.v0", {}, [array2d, array2d], [array2d, array2d]),
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
    if "LSTM" in name:
        model = with_padded(model)
    model.initialize(in_data, out_data)
    Y, backprop = model(in_data, is_train=True)
    assert_data_match(Y, out_data)
    dX = backprop(Y)
    assert_data_match(dX, in_data)
