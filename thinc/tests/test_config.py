import catalogue
from thinc.api import Config, registry, Model
import numpy


class my_registry(registry):
    cats = catalogue.create("thinc", "tests", "cats", entry_points=False)


def test_make_config_positional_args_dicts():
    cfg = {
        "hyper_params": {"n_hidden": 512, "dropout": 0.2, "learn_rate": 0.001},
        "model": {
            "@layers": "chain.v1",
            "*": {
                "relu1": {"@layers": "Relu.v1", "nO": 512, "dropout": 0.2},
                "relu2": {"@layers": "Relu.v1", "nO": 512, "dropout": 0.2},
                "softmax": {"@layers": "Softmax.v1"},
            },
        },
        "optimizer": {"@optimizers": "Adam.v1", "learn_rate": 0.001},
    }
    resolved = my_registry.resolve(cfg)
    model = resolved["model"]
    X = numpy.ones((784, 1), dtype="f")
    model.initialize(X=X, Y=numpy.zeros((784, 1), dtype="f"))
    model.begin_update(X)
    model.finish_update(resolved["optimizer"])


def test_handle_generic_model_type():
    """Test that validation can handle checks against arbitrary generic
    types in function argument annotations."""

    @my_registry.layers("my_transform.v1")
    def my_transform(model: Model[int, int]):
        model.name = "transformed_model"
        return model

    cfg = {"@layers": "my_transform.v1", "model": {"@layers": "Linear.v1"}}
    model = my_registry.resolve({"test": cfg})["test"]
    assert isinstance(model, Model)
    assert model.name == "transformed_model"


def test_arg_order_is_preserved():
    str_cfg = """
    [model]

    [model.chain]
    @layers = "chain.v1"

    [model.chain.*.hashembed]
    @layers = "HashEmbed.v1"
    nO = 8
    nV = 8

    [model.chain.*.expand_window]
    @layers = "expand_window.v1"
    window_size = 1
    """

    cfg = Config().from_str(str_cfg)
    resolved = my_registry.resolve(cfg)
    model = resolved["model"]["chain"]

    # Fails when arguments are sorted, because expand_window
    # is sorted before hashembed.
    assert model.name == "hashembed>>expand_window"
