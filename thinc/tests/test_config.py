import pytest
from typing import Iterable, Union, Optional, List, Callable, Dict
from types import GeneratorType
from pydantic import BaseModel, StrictBool, StrictFloat, PositiveInt, constr
import catalogue
import thinc.config
from thinc.config import ConfigValidationError
from thinc.types import Generator
from thinc.api import Config, RAdam, Model, NumpyOps
from thinc.util import partial
import numpy
import inspect

from .util import make_tempdir


EXAMPLE_CONFIG = """
[optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
use_averages = true

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.1
warmup_steps = 10000
total_steps = 100000

[pipeline]

[pipeline.parser]
name = "parser"
factory = "parser"

[pipeline.parser.model]
@layers = "spacy.ParserModel.v1"
hidden_depth = 1
hidden_width = 64
token_vector_width = 128

[pipeline.parser.model.tok2vec]
@layers = "Tok2Vec.v1"
width = ${pipeline.parser.model:token_vector_width}

[pipeline.parser.model.tok2vec.embed]
@layers = "spacy.MultiFeatureHashEmbed.v1"
width = ${pipeline.parser.model.tok2vec:width}

[pipeline.parser.model.tok2vec.embed.hidden]
@layers = "MLP.v1"
depth = 1
pieces = 3
layer_norm = true
outputs = ${pipeline.parser.model.tok2vec.embed:width}

[pipeline.parser.model.tok2vec.encode]
@layers = "spacy.MaxoutWindowEncoder.v1"
depth = 4
pieces = 3
window_size = 1

[pipeline.parser.model.lower]
@layers = "spacy.ParserLower.v1"

[pipeline.parser.model.upper]
@layers = "thinc.Linear.v1"
"""

OPTIMIZER_CFG = """
[optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
use_averages = true

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.1
warmup_steps = 10000
total_steps = 100000
"""


class my_registry(thinc.config.registry):
    cats = catalogue.create("thinc", "tests", "cats", entry_points=False)


class HelloIntsSchema(BaseModel):
    hello: int
    world: int

    class Config:
        extra = "forbid"


class DefaultsSchema(BaseModel):
    required: int
    optional: str = "default value"

    class Config:
        extra = "forbid"


class ComplexSchema(BaseModel):
    outer_req: int
    outer_opt: str = "default value"

    level2_req: HelloIntsSchema
    level2_opt: DefaultsSchema = DefaultsSchema(required=1)


@my_registry.cats.register("catsie.v1")
def catsie_v1(evil: StrictBool, cute: bool = True) -> str:
    if evil:
        return "scratch!"
    else:
        return "meow"


@my_registry.cats.register("catsie.v2")
def catsie_v2(evil: StrictBool, cute: bool = True, cute_level: int = 1) -> str:
    if evil:
        return "scratch!"
    else:
        if cute_level > 2:
            return "meow <3"
        return "meow"


good_catsie = {"@cats": "catsie.v1", "evil": False, "cute": True}
ok_catsie = {"@cats": "catsie.v1", "evil": False, "cute": False}
bad_catsie = {"@cats": "catsie.v1", "evil": True, "cute": True}
worst_catsie = {"@cats": "catsie.v1", "evil": True, "cute": False}


def test_validate_simple_config():
    simple_config = {"hello": 1, "world": 2}
    f, _, v = my_registry._fill(simple_config, HelloIntsSchema)
    assert f == simple_config
    assert v == simple_config


def test_invalidate_simple_config():
    invalid_config = {"hello": 1, "world": "hi!"}
    with pytest.raises(ConfigValidationError):
        my_registry._fill(invalid_config, HelloIntsSchema)


def test_invalidate_extra_args():
    invalid_config = {"hello": 1, "world": 2, "extra": 3}
    with pytest.raises(ConfigValidationError):
        my_registry._fill(invalid_config, HelloIntsSchema)


def test_fill_defaults_simple_config():
    valid_config = {"required": 1}
    filled, _, v = my_registry._fill(valid_config, DefaultsSchema)
    assert filled["required"] == 1
    assert filled["optional"] == "default value"
    invalid_config = {"optional": "some value"}
    with pytest.raises(ConfigValidationError):
        my_registry._fill(invalid_config, DefaultsSchema)


def test_fill_recursive_config():
    valid_config = {"outer_req": 1, "level2_req": {"hello": 4, "world": 7}}
    filled, _, validation = my_registry._fill(valid_config, ComplexSchema)
    assert filled["outer_req"] == 1
    assert filled["outer_opt"] == "default value"
    assert filled["level2_req"]["hello"] == 4
    assert filled["level2_req"]["world"] == 7
    assert filled["level2_opt"]["required"] == 1
    assert filled["level2_opt"]["optional"] == "default value"


def test_is_promise():
    assert my_registry.is_promise(good_catsie)
    assert not my_registry.is_promise({"hello": "world"})
    assert not my_registry.is_promise(1)
    invalid = {"@complex": "complex.v1", "rate": 1.0, "@cats": "catsie.v1"}
    assert my_registry.is_promise(invalid)


def test_get_constructor():
    func = my_registry.get_constructor(good_catsie)
    assert func is catsie_v1


def test_parse_args():
    args, kwargs = my_registry.parse_args(bad_catsie)
    assert args == []
    assert kwargs == {"evil": True, "cute": True}


def test_make_promise_schema():
    schema = my_registry.make_promise_schema(good_catsie)
    assert "evil" in schema.__fields__
    assert "cute" in schema.__fields__


def test_validate_promise():
    config = {"required": 1, "optional": good_catsie}
    filled, _, validated = my_registry._fill(config, DefaultsSchema)
    assert filled == config
    assert validated == {"required": 1, "optional": "meow"}


def test_fill_validate_promise():
    config = {"required": 1, "optional": {"@cats": "catsie.v1", "evil": False}}
    filled, _, validated = my_registry._fill(config, DefaultsSchema)
    assert filled["optional"]["cute"] is True


def test_fill_invalidate_promise():
    config = {"required": 1, "optional": {"@cats": "catsie.v1", "evil": False}}
    with pytest.raises(ConfigValidationError):
        my_registry._fill(config, HelloIntsSchema)
    config["optional"]["whiskers"] = True
    with pytest.raises(ConfigValidationError):
        my_registry._fill(config, DefaultsSchema)


def test_create_registry():
    with pytest.raises(ValueError):
        my_registry.create("cats")
    my_registry.create("dogs")
    assert hasattr(my_registry, "dogs")
    assert len(my_registry.dogs.get_all()) == 0
    my_registry.dogs.register("good_boy.v1", func=lambda x: x)
    assert len(my_registry.dogs.get_all()) == 1
    with pytest.raises(ValueError):
        my_registry.create("dogs")


def test_registry_methods():
    with pytest.raises(ValueError):
        my_registry.get("dfkoofkds", "catsie.v1")
    my_registry.cats.register("catsie.v123")(None)
    with pytest.raises(ValueError):
        my_registry.get("cats", "catsie.v123")


def test_make_from_config_no_schema():
    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": True}}}
    result = my_registry.make_from_config(config)
    assert result["one"] == 1
    assert result["two"] == {"three": "scratch!"}
    with pytest.raises(ConfigValidationError):
        config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": "true"}}}
        my_registry.make_from_config(config)


def test_make_from_config_schema():
    class TestBaseSubSchema(BaseModel):
        three: str

    class TestBaseSchema(BaseModel):
        one: PositiveInt
        two: TestBaseSubSchema

        class Config:
            extra = "forbid"

    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": True}}}
    my_registry.make_from_config(config, schema=TestBaseSchema)
    config = {"one": -1, "two": {"three": {"@cats": "catsie.v1", "evil": True}}}
    with pytest.raises(ConfigValidationError):
        # "one" is not a positive int
        my_registry.make_from_config(config, schema=TestBaseSchema)
    config = {"one": 1, "two": {"four": {"@cats": "catsie.v1", "evil": True}}}
    with pytest.raises(ConfigValidationError):
        # "three" is required in subschema
        my_registry.make_from_config(config, schema=TestBaseSchema)


def test_make_from_config_schema_coerced():
    class TestBaseSchema(BaseModel):
        test1: str
        test2: bool
        test3: float

    config = {"test1": 123, "test2": 1, "test3": 5}
    result = my_registry.make_from_config(config, schema=TestBaseSchema)
    assert result["test1"] == "123"
    assert result["test2"] is True
    assert result["test3"] == 5.0


def test_read_config():
    byte_string = EXAMPLE_CONFIG.encode("utf8")
    cfg = Config().from_bytes(byte_string)

    assert cfg["optimizer"]["beta1"] == 0.9
    assert cfg["optimizer"]["learn_rate"]["initial_rate"] == 0.1
    assert cfg["pipeline"]["parser"]["factory"] == "parser"
    assert cfg["pipeline"]["parser"]["model"]["tok2vec"]["width"] == 128


def test_optimizer_config():
    cfg = Config().from_str(OPTIMIZER_CFG)
    result = my_registry.make_from_config(cfg, validate=True)
    optimizer = result["optimizer"]
    assert optimizer.b1 == 0.9


def test_config_to_str():
    cfg = Config().from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()
    cfg = Config({"optimizer": {"foo": "bar"}}).from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_config_roundtrip_bytes():
    cfg = Config().from_str(OPTIMIZER_CFG)
    cfg_bytes = cfg.to_bytes()
    new_cfg = Config().from_bytes(cfg_bytes)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_config_roundtrip_disk():
    cfg = Config().from_str(OPTIMIZER_CFG)
    with make_tempdir() as path:
        cfg_path = path / "config.cfg"
        cfg.to_disk(cfg_path)
        new_cfg = Config().from_disk(cfg_path)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_validation_custom_types():
    def complex_args(
        rate: StrictFloat,
        steps: PositiveInt = 10,  # type: ignore
        log_level: constr(regex="(DEUG|INFO|WARNING|ERROR)") = "ERROR",
    ):
        return None

    my_registry.create("complex")
    my_registry.complex("complex.v1")(complex_args)
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "INFO"}
    my_registry.make_from_config({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": -1, "log_level": "INFO"}
    with pytest.raises(ConfigValidationError):
        # steps is not a positive int
        my_registry.make_from_config({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "none"}
    with pytest.raises(ConfigValidationError):
        # log_level is not a string matching the regex
        my_registry.make_from_config({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "INFO"}
    with pytest.raises(ConfigValidationError):
        # top-level object is promise
        my_registry.make_from_config(cfg)
    with pytest.raises(ConfigValidationError):
        # top-level object is promise
        my_registry.fill_config(cfg)
    cfg = {"@complex": "complex.v1", "rate": 1.0, "@cats": "catsie.v1"}
    with pytest.raises(ConfigValidationError):
        # two constructors
        my_registry.make_from_config({"config": cfg})


def test_validation_no_validate():
    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": "false"}}}
    result = my_registry.make_from_config(config, validate=False)
    assert result["one"] == 1
    assert result["two"] == {"three": "scratch!"}


def test_validation_fill_defaults():
    config = {"one": 1, "two": {"@cats": "catsie.v1", "evil": "hello"}}
    result = my_registry.fill_config(config, validate=False)
    assert len(result["two"]) == 3
    with pytest.raises(ConfigValidationError):
        # Required arg "evil" is not defined
        my_registry.fill_config(config)
    config = {"one": 1, "two": {"@cats": "catsie.v2", "evil": False}}
    # Fill in with new defaults
    result = my_registry.fill_config(config)
    assert len(result["two"]) == 4
    assert result["two"]["evil"] is False
    assert result["two"]["cute"] is True
    assert result["two"]["cute_level"] == 1


def test_make_config_positional_args():
    @my_registry.cats("catsie.v567")
    def catsie_567(*args: Optional[str], foo: str = "bar"):
        assert args[0] == "^_^"
        assert args[1] == "^(*.*)^"
        assert foo == "baz"
        return args[0]

    args = ["^_^", "^(*.*)^"]
    cfg = {"config": {"@cats": "catsie.v567", "foo": "baz", "*": args}}
    filled_cfg = my_registry.make_from_config(cfg)
    assert filled_cfg["config"] == "^_^"


def test_make_config_positional_args_complex():
    @my_registry.cats("catsie.v890")
    def catsie_890(*args: Optional[Union[StrictBool, PositiveInt]]):
        assert args[0] == 123
        return args[0]

    cfg = {"config": {"@cats": "catsie.v890", "*": [123, True, 1, False]}}
    filled_cfg = my_registry.make_from_config(cfg)
    assert filled_cfg["config"] == 123
    cfg = {"config": {"@cats": "catsie.v890", "*": [123, "True"]}}
    with pytest.raises(ConfigValidationError):
        # "True" is not a valid boolean or positive int
        my_registry.make_from_config(cfg)


def test_positional_args_to_from_string():
    cfg = """[a]\nb = 1\n* = ["foo","bar"]"""
    assert Config().from_str(cfg).to_str() == cfg
    cfg = """[a]\nb = 1\n\n[a.*.foo]\ntest = 1\n\n[a.*.bar]\ntest = 2"""
    assert Config().from_str(cfg).to_str() == cfg

    @my_registry.cats("catsie.v666")
    def catsie_666(*args, meow=False):
        return args

    cfg = """[a]\n@cats = "catsie.v666"\n* = ["foo","bar"]"""
    filled = my_registry.fill_config(Config().from_str(cfg)).to_str()
    assert filled == """[a]\n@cats = "catsie.v666"\n* = ["foo","bar"]\nmeow = false"""
    assert my_registry.make_from_config(Config().from_str(cfg)) == {"a": ("foo", "bar")}
    cfg = """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\nx = 1"""
    filled = my_registry.fill_config(Config().from_str(cfg)).to_str()
    assert filled == """[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\nx = 1"""
    assert my_registry.make_from_config(Config().from_str(cfg)) == {"a": ({"x": 1},)}

    @my_registry.cats("catsie.v777")
    def catsie_777(y: int = 1):
        return "meow" * y

    cfg = """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777\""""
    filled = my_registry.fill_config(Config().from_str(cfg)).to_str()
    expected = """[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 1"""
    assert filled == expected
    cfg = """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 3"""
    result = my_registry.make_from_config(Config().from_str(cfg))
    assert result == {"a": ("meowmeowmeow",)}


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
    loaded = my_registry.make_from_config(cfg)
    model = loaded["model"]
    X = numpy.ones((784, 1), dtype="f")
    model.initialize(X=X, Y=numpy.zeros((784, 1), dtype="f"))
    model.begin_update(X)
    model.finish_update(loaded["optimizer"])


def test_validation_generators_iterable():
    @my_registry.optimizers("test_optimizer.v1")
    def test_optimizer_v1(rate: float,) -> None:
        return None

    @my_registry.schedules("test_schedule.v1")
    def test_schedule_v1(some_value: float = 1.0) -> Iterable[float]:
        while True:
            yield some_value

    config = {"optimizer": {"@optimizers": "test_optimizer.v1", "rate": 0.1}}
    my_registry.make_from_config(config)


def test_validation_unset_type_hints():
    """Test that unset type hints are handled correctly (and treated as Any)."""

    @my_registry.optimizers("test_optimizer.v2")
    def test_optimizer_v2(rate, steps: int = 10) -> None:
        return None

    config = {"test": {"@optimizers": "test_optimizer.v2", "rate": 0.1, "steps": 20}}
    my_registry.make_from_config(config)


def test_validation_bad_function():
    @my_registry.optimizers("bad.v1")
    def bad() -> None:
        raise ValueError("This is an error in the function")
        return None

    @my_registry.optimizers("good.v1")
    def good() -> None:
        return None

    # Bad function
    config = {"test": {"@optimizers": "bad.v1"}}
    with pytest.raises(ConfigValidationError):
        my_registry.make_from_config(config)
    # Bad function call
    config = {"test": {"@optimizers": "good.v1", "invalid_arg": 1}}
    with pytest.raises(ConfigValidationError):
        my_registry.make_from_config(config)


def test_objects_from_config():
    config = {
        "optimizer": {
            "@optimizers": "my_cool_optimizer.v1",
            "beta1": 0.2,
            "learn_rate": {
                "@schedules": "my_cool_repetitive_schedule.v1",
                "base_rate": 0.001,
                "repeat": 4,
            },
        }
    }

    @thinc.registry.optimizers.register("my_cool_optimizer.v1")
    def make_my_optimizer(learn_rate: List[float], beta1: float):
        return RAdam(learn_rate, beta1=beta1)

    @thinc.registry.schedules("my_cool_repetitive_schedule.v1")
    def decaying(base_rate: float, repeat: int) -> List[float]:
        return repeat * [base_rate]

    loaded = my_registry.make_from_config(config)
    optimizer = loaded["optimizer"]
    assert optimizer.b1 == 0.2
    assert "learn_rate" in optimizer.schedules
    assert optimizer.learn_rate == 0.001


def test_partials_from_config():
    """Test that functions registered with partial applications are handled
    correctly (e.g. initializers)."""
    name = "uniform_init.v1"
    cfg = {"test": {"@initializers": name, "lo": -0.2}}
    func = my_registry.make_from_config(cfg)["test"]
    assert hasattr(func, "__call__")
    # The partial will still have lo as an arg, just with default
    assert len(inspect.signature(func).parameters) == 4
    # Make sure returned partial function has correct value set
    assert inspect.signature(func).parameters["lo"].default == -0.2
    # Actually call the function and verify
    func(NumpyOps(), (2, 3))
    # Make sure validation still works
    bad_cfg = {"test": {"@initializers": name, "lo": [0.5]}}
    with pytest.raises(ConfigValidationError):
        my_registry.make_from_config(bad_cfg)
    bad_cfg = {"test": {"@initializers": name, "lo": -0.2, "other": 10}}
    with pytest.raises(ConfigValidationError):
        my_registry.make_from_config(bad_cfg)


def test_partials_from_config_nested():
    """Test that partial functions are passed correctly to other registered
    functions that consume them (e.g. initializers -> layers)."""

    def test_initializer(a: int, b: int = 1) -> int:
        return a * b

    @my_registry.initializers("test_initializer.v1")
    def configure_test_initializer(b: int = 1) -> Callable[[int], int]:
        return partial(test_initializer, b=b)

    @my_registry.layers("test_layer.v1")
    def test_layer(init: Callable[[int], int], c: int = 1) -> Callable[[int], int]:
        return lambda x: x + init(c)

    cfg = {
        "@layers": "test_layer.v1",
        "c": 5,
        "init": {"@initializers": "test_initializer.v1", "b": 10},
    }
    func = my_registry.make_from_config({"test": cfg})["test"]
    assert func(1) == 51
    assert func(100) == 150


def test_validate_generator():
    """Test that generator replacement for validation in config doesn't
    actually replace the returned value."""

    @my_registry.schedules("test_schedule.v2")
    def test_schedule():
        while True:
            yield 10

    cfg = {"@schedules": "test_schedule.v2"}
    result = my_registry.make_from_config({"test": cfg})["test"]
    assert isinstance(result, GeneratorType)

    @my_registry.optimizers("test_optimizer.v2")
    def test_optimizer2(rate: Generator) -> Generator:
        return rate

    cfg = {
        "@optimizers": "test_optimizer.v2",
        "rate": {"@schedules": "test_schedule.v2"},
    }
    result = my_registry.make_from_config({"test": cfg})["test"]
    assert isinstance(result, GeneratorType)

    @my_registry.optimizers("test_optimizer.v3")
    def test_optimizer3(schedules: Dict[str, Generator]) -> Generator:
        return schedules["rate"]

    cfg = {
        "@optimizers": "test_optimizer.v3",
        "schedules": {"rate": {"@schedules": "test_schedule.v2"}},
    }
    result = my_registry.make_from_config({"test": cfg})["test"]
    assert isinstance(result, GeneratorType)

    @my_registry.optimizers("test_optimizer.v4")
    def test_optimizer4(*schedules: Generator) -> Generator:
        return schedules[0]


def test_handle_generic_model_type():
    """Test that validation can handle checks against arbitrary generic
    types in function argument annotations."""

    @my_registry.layers("my_transform.v1")
    def my_transform(model: Model[int, int]):
        model.name = "transformed_model"
        return model

    cfg = {"@layers": "my_transform.v1", "model": {"@layers": "Linear.v1"}}
    model = my_registry.make_from_config({"test": cfg})["test"]
    assert isinstance(model, Model)
    assert model.name == "transformed_model"


@pytest.mark.parametrize(
    "cfg",
    [
        "[a]\nb = 1\nc = 2\n\n[a.c]\nd = 3",
        "[a]\nb = 1\n\n[a.c]\nd = 2\n\n[a.c.d]\ne = 3",
    ],
)
def test_handle_error_duplicate_keys(cfg):
    """This would cause very cryptic error when interpreting config.
    (TypeError: 'X' object does not support item assignment)
    """
    with pytest.raises(ConfigValidationError):
        Config().from_str(cfg)
