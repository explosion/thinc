from functools import partial
from thinc.api import Model, chain, Linear


def test_issue208():
    """Test issue that was caused by trying to flatten nested chains."""
    layer1 = Linear(nO=9, nI=3)
    layer2 = Linear(nO=12, nI=9)
    layer3 = Linear(nO=5, nI=12)
    with Model.define_operators({">>": chain}):
        model = layer1 >> layer2 >> layer3
    assert len(model.layers) == 2
    model.set_dim("nO", 5)
