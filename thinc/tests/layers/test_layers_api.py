from thinc.api import registry
import pytest


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("CauchySimilarity.v0", {}),
        ("Dropout.v0", {}),
        ("Embed.v0", {}),
        ("ExtractWindow.v0", {}),
        ("FeatureExtractor.v0", {"columns": [1, 2]}),
        ("HashEmbed.v0", {"nO": 1, "nV": 2}),
        ("LayerNorm.v0", {}),
        ("Linear.v0", {}),
        ("BiLSTM.v0", {}),
        ("LSTM.v0", {}),
        ("Maxout.v0", {}),
        ("MaxPool.v0", {}),
        ("MeanPool.v0", {}),
        ("Mish.v0", {}),
        ("MultiSoftmax.v0", {"nOs": (1, 2, 3)}),
        ("ParametricAttention.v0", {}),
        ("ReLu.v0", {}),
        ("Softmax.v0", {}),
        ("SparseLinear.v0", {}),
        ("StaticVectors.v0", {"lang": "en", "nO": 5}),
        ("SumPool.v0", {}),
    ],
)
def test_layers_from_config(name, kwargs):
    cfg = {"@layers": name, **kwargs}
    filled = registry.fill_config({"config": cfg})
    registry.make_from_config(filled)
