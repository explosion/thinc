from functools import partial
from thinc.api import Model, chain, Linear


def test_issue208():
    """Test bug that caused uncalled initializers in nested chain."""
    init_was_called = {}
    def register_init(name, model, X=None, Y=None):
        init_was_called[name] = True

    model1 = Linear(5)
    model2 = Linear(5)
    model3 = Linear(5)
    model1._init = partial(register_init, "one")
    model2._init = partial(register_init, "two")
    model3._init = partial(register_init, "three")

    with Model.define_operators({">>": chain}):
        model = model1 >> model2 >> model3
    
    assert not init_was_called
    model.initialize()
    assert init_was_called["one"]
    assert init_was_called["two"]
    assert init_was_called["three"]
