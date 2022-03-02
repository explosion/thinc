import pytest
from typing import Iterable, Union, Optional, List, Callable, Dict, Any
from types import GeneratorType
from pydantic import BaseModel, StrictBool, StrictFloat, PositiveInt, constr
import catalogue
import thinc.config
from thinc.config import ConfigValidationError
from thinc.types import Generator, Ragged
from thinc.api import Config, RAdam, Model, NumpyOps
from thinc.util import partial
import numpy
import inspect
import pickle

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
    with pytest.raises(ConfigValidationError) as exc_info:
        my_registry._fill(invalid_config, HelloIntsSchema)
    error = exc_info.value
    assert len(error.errors) == 1
    assert "type_error.integer" in error.error_types


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
    my_registry.get_constructor(good_catsie) == ("cats", "catsie.v1")


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


def test_resolve_no_schema():
    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": True}}}
    result = my_registry.resolve({"cfg": config})["cfg"]
    assert result["one"] == 1
    assert result["two"] == {"three": "scratch!"}
    with pytest.raises(ConfigValidationError):
        config = {"two": {"three": {"@cats": "catsie.v1", "evil": "true"}}}
        my_registry.resolve(config)


def test_resolve_schema():
    class TestBaseSubSchema(BaseModel):
        three: str

    class TestBaseSchema(BaseModel):
        one: PositiveInt
        two: TestBaseSubSchema

        class Config:
            extra = "forbid"

    class TestSchema(BaseModel):
        cfg: TestBaseSchema

    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": True}}}
    my_registry.resolve({"cfg": config}, schema=TestSchema)
    config = {"one": -1, "two": {"three": {"@cats": "catsie.v1", "evil": True}}}
    with pytest.raises(ConfigValidationError):
        # "one" is not a positive int
        my_registry.resolve({"cfg": config}, schema=TestSchema)
    config = {"one": 1, "two": {"four": {"@cats": "catsie.v1", "evil": True}}}
    with pytest.raises(ConfigValidationError):
        # "three" is required in subschema
        my_registry.resolve({"cfg": config}, schema=TestSchema)


def test_resolve_schema_coerced():
    class TestBaseSchema(BaseModel):
        test1: str
        test2: bool
        test3: float

    class TestSchema(BaseModel):
        cfg: TestBaseSchema

    config = {"test1": 123, "test2": 1, "test3": 5}
    filled = my_registry.fill({"cfg": config}, schema=TestSchema)
    result = my_registry.resolve({"cfg": config}, schema=TestSchema)
    assert result["cfg"] == {"test1": "123", "test2": True, "test3": 5.0}
    # This only affects the resolved config, not the filled config
    assert filled["cfg"] == config


def test_read_config():
    byte_string = EXAMPLE_CONFIG.encode("utf8")
    cfg = Config().from_bytes(byte_string)

    assert cfg["optimizer"]["beta1"] == 0.9
    assert cfg["optimizer"]["learn_rate"]["initial_rate"] == 0.1
    assert cfg["pipeline"]["parser"]["factory"] == "parser"
    assert cfg["pipeline"]["parser"]["model"]["tok2vec"]["width"] == 128


def test_optimizer_config():
    cfg = Config().from_str(OPTIMIZER_CFG)
    optimizer = my_registry.resolve(cfg, validate=True)["optimizer"]
    assert optimizer.b1 == 0.9


def test_config_to_str():
    cfg = Config().from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()
    cfg = Config({"optimizer": {"foo": "bar"}}).from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_config_to_str_creates_intermediate_blocks():
    cfg = Config({"optimizer": {"foo": {"bar": 1}}})
    assert (
        cfg.to_str().strip()
        == """
[optimizer]

[optimizer.foo]
bar = 1
    """.strip()
    )


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


def test_config_roundtrip_disk_respects_path_subclasses(pathy_fixture):
    cfg = Config().from_str(OPTIMIZER_CFG)
    cfg_path = pathy_fixture / "config.cfg"
    cfg.to_disk(cfg_path)
    new_cfg = Config().from_disk(cfg_path)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_config_to_str_invalid_defaults():
    """Test that an error is raised if a config contains top-level keys without
    a section that would otherwise be interpreted as [DEFAULT] (which causes
    the values to be included in *all* other sections).
    """
    cfg = {"one": 1, "two": {"@cats": "catsie.v1", "evil": "hello"}}
    with pytest.raises(ConfigValidationError):
        Config(cfg).to_str()
    config_str = "[DEFAULT]\none = 1"
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)


def test_validation_custom_types():
    def complex_args(
        rate: StrictFloat,
        steps: PositiveInt = 10,  # type: ignore
        log_level: constr(regex="(DEBUG|INFO|WARNING|ERROR)") = "ERROR",
    ):
        return None

    my_registry.create("complex")
    my_registry.complex("complex.v1")(complex_args)
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "INFO"}
    my_registry.resolve({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": -1, "log_level": "INFO"}
    with pytest.raises(ConfigValidationError):
        # steps is not a positive int
        my_registry.resolve({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "none"}
    with pytest.raises(ConfigValidationError):
        # log_level is not a string matching the regex
        my_registry.resolve({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "INFO"}
    with pytest.raises(ConfigValidationError):
        # top-level object is promise
        my_registry.resolve(cfg)
    with pytest.raises(ConfigValidationError):
        # top-level object is promise
        my_registry.fill(cfg)
    cfg = {"@complex": "complex.v1", "rate": 1.0, "@cats": "catsie.v1"}
    with pytest.raises(ConfigValidationError):
        # two constructors
        my_registry.resolve({"config": cfg})


def test_validation_no_validate():
    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": "false"}}}
    result = my_registry.resolve({"cfg": config}, validate=False)
    filled = my_registry.fill({"cfg": config}, validate=False)
    assert result["cfg"]["one"] == 1
    assert result["cfg"]["two"] == {"three": "scratch!"}
    assert filled["cfg"]["two"]["three"]["evil"] == "false"
    assert filled["cfg"]["two"]["three"]["cute"] is True


def test_validation_fill_defaults():
    config = {"cfg": {"one": 1, "two": {"@cats": "catsie.v1", "evil": "hello"}}}
    result = my_registry.fill(config, validate=False)
    assert len(result["cfg"]["two"]) == 3
    with pytest.raises(ConfigValidationError):
        # Required arg "evil" is not defined
        my_registry.fill(config)
    config = {"cfg": {"one": 1, "two": {"@cats": "catsie.v2", "evil": False}}}
    # Fill in with new defaults
    result = my_registry.fill(config)
    assert len(result["cfg"]["two"]) == 4
    assert result["cfg"]["two"]["evil"] is False
    assert result["cfg"]["two"]["cute"] is True
    assert result["cfg"]["two"]["cute_level"] == 1


def test_make_config_positional_args():
    @my_registry.cats("catsie.v567")
    def catsie_567(*args: Optional[str], foo: str = "bar"):
        assert args[0] == "^_^"
        assert args[1] == "^(*.*)^"
        assert foo == "baz"
        return args[0]

    args = ["^_^", "^(*.*)^"]
    cfg = {"config": {"@cats": "catsie.v567", "foo": "baz", "*": args}}
    assert my_registry.resolve(cfg)["config"] == "^_^"


def test_make_config_positional_args_complex():
    @my_registry.cats("catsie.v890")
    def catsie_890(*args: Optional[Union[StrictBool, PositiveInt]]):
        assert args[0] == 123
        return args[0]

    cfg = {"config": {"@cats": "catsie.v890", "*": [123, True, 1, False]}}
    assert my_registry.resolve(cfg)["config"] == 123
    cfg = {"config": {"@cats": "catsie.v890", "*": [123, "True"]}}
    with pytest.raises(ConfigValidationError):
        # "True" is not a valid boolean or positive int
        my_registry.resolve(cfg)


def test_positional_args_to_from_string():
    cfg = """[a]\nb = 1\n* = ["foo","bar"]"""
    assert Config().from_str(cfg).to_str() == cfg
    cfg = """[a]\nb = 1\n\n[a.*.bar]\ntest = 2\n\n[a.*.foo]\ntest = 1"""
    assert Config().from_str(cfg).to_str() == cfg

    @my_registry.cats("catsie.v666")
    def catsie_666(*args, meow=False):
        return args

    cfg = """[a]\n@cats = "catsie.v666"\n* = ["foo","bar"]"""
    filled = my_registry.fill(Config().from_str(cfg)).to_str()
    assert filled == """[a]\n@cats = "catsie.v666"\n* = ["foo","bar"]\nmeow = false"""
    resolved = my_registry.resolve(Config().from_str(cfg))
    assert resolved == {"a": ("foo", "bar")}
    cfg = """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\nx = 1"""
    filled = my_registry.fill(Config().from_str(cfg)).to_str()
    assert filled == """[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\nx = 1"""
    resolved = my_registry.resolve(Config().from_str(cfg))
    assert resolved == {"a": ({"x": 1},)}

    @my_registry.cats("catsie.v777")
    def catsie_777(y: int = 1):
        return "meow" * y

    cfg = """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777\""""
    filled = my_registry.fill(Config().from_str(cfg)).to_str()
    expected = """[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 1"""
    assert filled == expected
    cfg = """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 3"""
    result = my_registry.resolve(Config().from_str(cfg))
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
    resolved = my_registry.resolve(cfg)
    model = resolved["model"]
    X = numpy.ones((784, 1), dtype="f")
    model.initialize(X=X, Y=numpy.zeros((784, 1), dtype="f"))
    model.begin_update(X)
    model.finish_update(resolved["optimizer"])


def test_validation_generators_iterable():
    @my_registry.optimizers("test_optimizer.v1")
    def test_optimizer_v1(rate: float) -> None:
        return None

    @my_registry.schedules("test_schedule.v1")
    def test_schedule_v1(some_value: float = 1.0) -> Iterable[float]:
        while True:
            yield some_value

    config = {"optimizer": {"@optimizers": "test_optimizer.v1", "rate": 0.1}}
    my_registry.resolve(config)


def test_validation_unset_type_hints():
    """Test that unset type hints are handled correctly (and treated as Any)."""

    @my_registry.optimizers("test_optimizer.v2")
    def test_optimizer_v2(rate, steps: int = 10) -> None:
        return None

    config = {"test": {"@optimizers": "test_optimizer.v2", "rate": 0.1, "steps": 20}}
    my_registry.resolve(config)


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
    with pytest.raises(ValueError):
        my_registry.resolve(config)
    # Bad function call
    config = {"test": {"@optimizers": "good.v1", "invalid_arg": 1}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(config)


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

    optimizer = my_registry.resolve(config)["optimizer"]
    assert optimizer.b1 == 0.2
    assert "learn_rate" in optimizer.schedules
    assert optimizer.learn_rate == 0.001


def test_partials_from_config():
    """Test that functions registered with partial applications are handled
    correctly (e.g. initializers)."""
    name = "uniform_init.v1"
    cfg = {"test": {"@initializers": name, "lo": -0.2}}
    func = my_registry.resolve(cfg)["test"]
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
        my_registry.resolve(bad_cfg)
    bad_cfg = {"test": {"@initializers": name, "lo": -0.2, "other": 10}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(bad_cfg)


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
    func = my_registry.resolve({"test": cfg})["test"]
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
    result = my_registry.resolve({"test": cfg})["test"]
    assert isinstance(result, GeneratorType)

    @my_registry.optimizers("test_optimizer.v2")
    def test_optimizer2(rate: Generator) -> Generator:
        return rate

    cfg = {
        "@optimizers": "test_optimizer.v2",
        "rate": {"@schedules": "test_schedule.v2"},
    }
    result = my_registry.resolve({"test": cfg})["test"]
    assert isinstance(result, GeneratorType)

    @my_registry.optimizers("test_optimizer.v3")
    def test_optimizer3(schedules: Dict[str, Generator]) -> Generator:
        return schedules["rate"]

    cfg = {
        "@optimizers": "test_optimizer.v3",
        "schedules": {"rate": {"@schedules": "test_schedule.v2"}},
    }
    result = my_registry.resolve({"test": cfg})["test"]
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
    model = my_registry.resolve({"test": cfg})["test"]
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


@pytest.mark.parametrize(
    "cfg,is_valid",
    [("[a]\nb = 1\n\n[a.c]\nd = 3", True), ("[a]\nb = 1\n\n[A.c]\nd = 2", False)],
)
def test_cant_expand_undefined_block(cfg, is_valid):
    """Test that you can't expand a block that hasn't been created yet. This
    comes up when you typo a name, and if we allow expansion of undefined blocks,
    it's very hard to create good errors for those typos.
    """
    if is_valid:
        Config().from_str(cfg)
    else:
        with pytest.raises(ConfigValidationError):
            Config().from_str(cfg)


def test_fill_config_overrides():
    config = {
        "cfg": {
            "one": 1,
            "two": {"three": {"@cats": "catsie.v1", "evil": True, "cute": False}},
        }
    }
    overrides = {"cfg.two.three.evil": False}
    result = my_registry.fill(config, overrides=overrides, validate=True)
    assert result["cfg"]["two"]["three"]["evil"] is False
    # Test that promises can be overwritten as well
    overrides = {"cfg.two.three": 3}
    result = my_registry.fill(config, overrides=overrides, validate=True)
    assert result["cfg"]["two"]["three"] == 3
    # Test that value can be overwritten with promises and that the result is
    # interpreted and filled correctly
    overrides = {"cfg": {"one": {"@cats": "catsie.v1", "evil": False}, "two": None}}
    result = my_registry.fill(config, overrides=overrides)
    assert result["cfg"]["two"] is None
    assert result["cfg"]["one"]["@cats"] == "catsie.v1"
    assert result["cfg"]["one"]["evil"] is False
    assert result["cfg"]["one"]["cute"] is True
    # Overwriting with wrong types should cause validation error
    with pytest.raises(ConfigValidationError):
        overrides = {"cfg.two.three.evil": 20}
        my_registry.fill(config, overrides=overrides, validate=True)
    # Overwriting with incomplete promises should cause validation error
    with pytest.raises(ConfigValidationError):
        overrides = {"cfg": {"one": {"@cats": "catsie.v1"}, "two": None}}
        my_registry.fill(config, overrides=overrides)
    # Overrides that don't match config should raise error
    with pytest.raises(ConfigValidationError):
        overrides = {"cfg.two.three.evil": False, "two.four": True}
        my_registry.fill(config, overrides=overrides, validate=True)
    with pytest.raises(ConfigValidationError):
        overrides = {"cfg.five": False}
        my_registry.fill(config, overrides=overrides, validate=True)


def test_resolve_overrides():
    config = {
        "cfg": {
            "one": 1,
            "two": {"three": {"@cats": "catsie.v1", "evil": True, "cute": False}},
        }
    }
    overrides = {"cfg.two.three.evil": False}
    result = my_registry.resolve(config, overrides=overrides, validate=True)
    assert result["cfg"]["two"]["three"] == "meow"
    # Test that promises can be overwritten as well
    overrides = {"cfg.two.three": 3}
    result = my_registry.resolve(config, overrides=overrides, validate=True)
    assert result["cfg"]["two"]["three"] == 3
    # Test that value can be overwritten with promises
    overrides = {"cfg": {"one": {"@cats": "catsie.v1", "evil": False}, "two": None}}
    result = my_registry.resolve(config, overrides=overrides)
    assert result["cfg"]["one"] == "meow"
    assert result["cfg"]["two"] is None
    # Overwriting with wrong types should cause validation error
    with pytest.raises(ConfigValidationError):
        overrides = {"cfg.two.three.evil": 20}
        my_registry.resolve(config, overrides=overrides, validate=True)
    # Overwriting with incomplete promises should cause validation error
    with pytest.raises(ConfigValidationError):
        overrides = {"cfg": {"one": {"@cats": "catsie.v1"}, "two": None}}
        my_registry.resolve(config, overrides=overrides)
    # Overrides that don't match config should raise error
    with pytest.raises(ConfigValidationError):
        overrides = {"cfg.two.three.evil": False, "cfg.two.four": True}
        my_registry.resolve(config, overrides=overrides, validate=True)
    with pytest.raises(ConfigValidationError):
        overrides = {"cfg.five": False}
        my_registry.resolve(config, overrides=overrides, validate=True)


@pytest.mark.parametrize(
    "prop,expected",
    [("a.b.c", True), ("a.b", True), ("a", True), ("a.e", True), ("a.b.c.d", False)],
)
def test_is_in_config(prop, expected):
    config = {"a": {"b": {"c": 5, "d": 6}, "e": [1, 2]}}
    assert my_registry._is_in_config(prop, config) is expected


def test_resolve_prefilled_values():
    class Language(object):
        def __init__(self):
            ...

    @my_registry.optimizers("prefilled.v1")
    def prefilled(nlp: Language, value: int = 10):
        return (nlp, value)

    # Passing an instance of Language here via the config is bad, since it
    # won't serialize to a string, but we still test for it
    config = {"test": {"@optimizers": "prefilled.v1", "nlp": Language(), "value": 50}}
    resolved = my_registry.resolve(config, validate=True)
    result = resolved["test"]
    assert isinstance(result[0], Language)
    assert result[1] == 50


def test_fill_config_dict_return_type():
    """Test that a registered function returning a dict is handled correctly."""

    @my_registry.cats.register("catsie_with_dict.v1")
    def catsie_with_dict(evil: StrictBool) -> Dict[str, bool]:
        return {"not_evil": not evil}

    config = {"test": {"@cats": "catsie_with_dict.v1", "evil": False}, "foo": 10}
    result = my_registry.fill({"cfg": config}, validate=True)["cfg"]["test"]
    assert result["evil"] is False
    assert "not_evil" not in result
    result = my_registry.resolve({"cfg": config}, validate=True)["cfg"]["test"]
    assert result["not_evil"] is True


def test_deepcopy_config():
    config = Config({"a": 1, "b": {"c": 2, "d": 3}})
    copied = config.copy()
    # Same values but not same object
    assert config == copied
    assert config is not copied
    # Check for error if value can't be pickled/deepcopied
    config = Config({"a": 1, "b": numpy})
    with pytest.raises(ValueError):
        config.copy()


def test_config_to_str_simple_promises():
    """Test that references to function registries without arguments are
    serialized inline as dict."""
    config_str = """[section]\nsubsection = {"@registry":"value"}"""
    config = Config().from_str(config_str)
    assert config["section"]["subsection"]["@registry"] == "value"
    assert config.to_str() == config_str


def test_config_from_str_invalid_section():
    config_str = """[a]\nb = null\n\n[a.b]\nc = 1"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)

    config_str = """[a]\nb = null\n\n[a.b.c]\nd = 1"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)


def test_config_to_str_order():
    """Test that Config.to_str orders the sections."""
    config = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": {"g": {"h": {"i": 4, "j": 5}}}}
    expected = (
        "[a]\ne = 3\n\n[a.b]\nc = 1\nd = 2\n\n[f]\n\n[f.g]\n\n[f.g.h]\ni = 4\nj = 5"
    )
    config = Config(config)
    assert config.to_str() == expected


@pytest.mark.parametrize("d", [".", ":"])
def test_config_interpolation(d):
    """Test that config values are interpolated correctly. The parametrized
    value is the final divider (${a.b} vs. ${a:b}). Both should now work and be
    valid. The double {{ }} in the config strings are required to prevent the
    references from being interpreted as an actual f-string variable.
    """
    c_str = """[a]\nfoo = "hello"\n\n[b]\nbar = ${foo}"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    c_str = f"""[a]\nfoo = "hello"\n\n[b]\nbar = ${{a{d}foo}}"""
    assert Config().from_str(c_str)["b"]["bar"] == "hello"
    c_str = f"""[a]\nfoo = "hello"\n\n[b]\nbar = ${{a{d}foo}}!"""
    assert Config().from_str(c_str)["b"]["bar"] == "hello!"
    c_str = f"""[a]\nfoo = "hello"\n\n[b]\nbar = "${{a{d}foo}}!\""""
    assert Config().from_str(c_str)["b"]["bar"] == "hello!"
    c_str = f"""[a]\nfoo = 15\n\n[b]\nbar = ${{a{d}foo}}!"""
    assert Config().from_str(c_str)["b"]["bar"] == "15!"
    c_str = f"""[a]\nfoo = ["x", "y"]\n\n[b]\nbar = ${{a{d}foo}}"""
    assert Config().from_str(c_str)["b"]["bar"] == ["x", "y"]
    # Interpolation within the same section
    c_str = f"""[a]\nfoo = "x"\nbar = ${{a{d}foo}}\nbaz = "${{a{d}foo}}y\""""
    assert Config().from_str(c_str)["a"]["bar"] == "x"
    assert Config().from_str(c_str)["a"]["baz"] == "xy"


def test_config_interpolation_lists():
    # Test that lists are preserved correctly
    c_str = """[a]\nb = 1\n\n[c]\nd = ["hello ${a.b}", "world"]"""
    config = Config().from_str(c_str, interpolate=False)
    assert config["c"]["d"] == ["hello ${a.b}", "world"]
    config = config.interpolate()
    assert config["c"]["d"] == ["hello 1", "world"]
    c_str = """[a]\nb = 1\n\n[c]\nd = [${a.b}, "hello ${a.b}", "world"]"""
    config = Config().from_str(c_str)
    assert config["c"]["d"] == [1, "hello 1", "world"]
    config = Config().from_str(c_str, interpolate=False)
    # NOTE: This currently doesn't work, because we can't know how to JSON-load
    # the uninterpolated list [${a.b}].
    # assert config["c"]["d"] == ["${a.b}", "hello ${a.b}", "world"]
    # config = config.interpolate()
    # assert config["c"]["d"] == [1, "hello 1", "world"]
    c_str = """[a]\nb = 1\n\n[c]\nd = ["hello", ${a}]"""
    config = Config().from_str(c_str)
    assert config["c"]["d"] == ["hello", {"b": 1}]
    c_str = """[a]\nb = 1\n\n[c]\nd = ["hello", "hello ${a}"]"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    config_str = """[a]\nb = 1\n\n[c]\nd = ["hello", {"x": ["hello ${a.b}"], "y": 2}]"""
    config = Config().from_str(config_str)
    assert config["c"]["d"] == ["hello", {"x": ["hello 1"], "y": 2}]
    config_str = """[a]\nb = 1\n\n[c]\nd = ["hello", {"x": [${a.b}], "y": 2}]"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)


@pytest.mark.parametrize("d", [".", ":"])
def test_config_interpolation_sections(d):
    """Test that config sections are interpolated correctly. The parametrized
    value is the final divider (${a.b} vs. ${a:b}). Both should now work and be
    valid. The double {{ }} in the config strings are required to prevent the
    references from being interpreted as an actual f-string variable.
    """
    # Simple block references
    c_str = """[a]\nfoo = "hello"\nbar = "world"\n\n[b]\nc = ${a}"""
    config = Config().from_str(c_str)
    assert config["b"]["c"] == config["a"]
    # References with non-string values
    c_str = f"""[a]\nfoo = "hello"\n\n[a.x]\ny = ${{a{d}b}}\n\n[a.b]\nc = 1\nd = [10]"""
    config = Config().from_str(c_str)
    assert config["a"]["x"]["y"] == config["a"]["b"]
    # Multiple references in the same string
    c_str = f"""[a]\nx = "string"\ny = 10\n\n[b]\nz = "${{a{d}x}}/${{a{d}y}}\""""
    config = Config().from_str(c_str)
    assert config["b"]["z"] == "string/10"
    # Non-string references in string (converted to string)
    c_str = f"""[a]\nx = ["hello", "world"]\n\n[b]\ny = "result: ${{a{d}x}}\""""
    config = Config().from_str(c_str)
    assert config["b"]["y"] == 'result: ["hello", "world"]'
    # References to sections referencing sections
    c_str = """[a]\nfoo = "x"\n\n[b]\nbar = ${a}\n\n[c]\nbaz = ${b}"""
    config = Config().from_str(c_str)
    assert config["b"]["bar"] == config["a"]
    assert config["c"]["baz"] == config["b"]
    # References to section values referencing other sections
    c_str = f"""[a]\nfoo = "x"\n\n[b]\nbar = ${{a}}\n\n[c]\nbaz = ${{b{d}bar}}"""
    config = Config().from_str(c_str)
    assert config["c"]["baz"] == config["b"]["bar"]
    # References to sections with subsections
    c_str = """[a]\nfoo = "x"\n\n[a.b]\nbar = 100\n\n[c]\nbaz = ${a}"""
    config = Config().from_str(c_str)
    assert config["c"]["baz"] == config["a"]
    # Infinite recursion
    c_str = """[a]\nfoo ="x"\n\n[a.b]\nbar = ${a}"""
    config = Config().from_str(c_str)
    assert config["a"]["b"]["bar"] == config["a"]
    c_str = f"""[a]\nfoo = "x"\n\n[b]\nbar = ${{a}}\n\n[c]\nbaz = ${{b.bar{d}foo}}"""
    # We can't reference not-yet interpolated subsections
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    # Generally invalid references
    c_str = f"""[a]\nfoo = ${{b{d}bar}}"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    # We can't reference sections or promises within strings
    c_str = """[a]\n\n[a.b]\nfoo = "x: ${c}"\n\n[c]\nbar = 1\nbaz = 2"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)


def test_config_from_str_overrides():
    config_str = """[a]\nb = 1\n\n[a.c]\nd = 2\ne = 3\n\n[f]\ng = {"x": "y"}"""
    # Basic value substitution
    overrides = {"a.b": 10, "a.c.d": 20}
    config = Config().from_str(config_str, overrides=overrides)
    assert config["a"]["b"] == 10
    assert config["a"]["c"]["d"] == 20
    assert config["a"]["c"]["e"] == 3
    # Valid values that previously weren't in config
    config = Config().from_str(config_str, overrides={"a.c.f": 100})
    assert config["a"]["c"]["d"] == 2
    assert config["a"]["c"]["e"] == 3
    assert config["a"]["c"]["f"] == 100
    # Invalid keys and sections
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str, overrides={"f": 10})
    # This currently isn't expected to work, because the dict in f.g is not
    # interpreted as a section while the config is still just the configparser
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str, overrides={"f.g.x": "z"})
    # With variables (values)
    config_str = """[a]\nb = 1\n\n[a.c]\nd = 2\ne = ${a:b}"""
    config = Config().from_str(config_str, overrides={"a.b": 10})
    assert config["a"]["b"] == 10
    assert config["a"]["c"]["e"] == 10
    # With variables (sections)
    config_str = """[a]\nb = 1\n\n[a.c]\nd = 2\n[e]\nf = ${a.c}"""
    config = Config().from_str(config_str, overrides={"a.c.d": 20})
    assert config["a"]["c"]["d"] == 20
    assert config["e"]["f"] == {"d": 20}


def test_config_reserved_aliases():
    """Test that the auto-generated pydantic schemas auto-alias reserved
    attributes like "validate" that would otherwise cause NameError."""

    @my_registry.cats("catsie.with_alias")
    def catsie_with_alias(validate: StrictBool = False):
        return validate

    cfg = {"@cats": "catsie.with_alias", "validate": True}
    resolved = my_registry.resolve({"test": cfg})
    filled = my_registry.fill({"test": cfg})
    assert resolved["test"] is True
    assert filled["test"] == cfg
    cfg = {"@cats": "catsie.with_alias", "validate": 20}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve({"test": cfg})


@pytest.mark.parametrize("d", [".", ":"])
def test_config_no_interpolation(d):
    """Test that interpolation is correctly preserved. The parametrized
    value is the final divider (${a.b} vs. ${a:b}). Both should now work and be
    valid. The double {{ }} in the config strings are required to prevent the
    references from being interpreted as an actual f-string variable.
    """
    c_str = f"""[a]\nb = 1\n\n[c]\nd = ${{a{d}b}}\ne = \"hello${{a{d}b}}"\nf = ${{a}}"""
    config = Config().from_str(c_str, interpolate=False)
    assert not config.is_interpolated
    assert config["c"]["d"] == f"${{a{d}b}}"
    assert config["c"]["e"] == f'"hello${{a{d}b}}"'
    assert config["c"]["f"] == "${a}"
    config2 = Config().from_str(config.to_str(), interpolate=True)
    assert config2.is_interpolated
    assert config2["c"]["d"] == 1
    assert config2["c"]["e"] == "hello1"
    assert config2["c"]["f"] == {"b": 1}
    config3 = config.interpolate()
    assert config3.is_interpolated
    assert config3["c"]["d"] == 1
    assert config3["c"]["e"] == "hello1"
    assert config3["c"]["f"] == {"b": 1}
    # Bad non-serializable value
    cfg = {"x": {"y": numpy.asarray([[1, 2], [4, 5]], dtype="f"), "z": f"${{x{d}y}}"}}
    with pytest.raises(ConfigValidationError):
        Config(cfg).interpolate()


def test_config_no_interpolation_registry():
    config_str = """[a]\nbad = true\n[b]\n@cats = "catsie.v1"\nevil = ${a:bad}\n\n[c]\n d = ${b}"""
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    assert config["b"]["evil"] == "${a:bad}"
    assert config["c"]["d"] == "${b}"
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved["b"] == "scratch!"
    assert resolved["c"]["d"] == "scratch!"
    assert filled["b"]["evil"] == "${a:bad}"
    assert filled["b"]["cute"] is True
    assert filled["c"]["d"] == "${b}"
    interpolated = filled.interpolate()
    assert interpolated.is_interpolated
    assert interpolated["b"]["evil"] is True
    assert interpolated["c"]["d"] == interpolated["b"]
    config = Config().from_str(config_str, interpolate=True)
    assert config.is_interpolated
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved["b"] == "scratch!"
    assert resolved["c"]["d"] == "scratch!"
    assert filled["b"]["evil"] is True
    assert filled["c"]["d"] == filled["b"]
    # Resolving a non-interpolated filled config
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    filled = my_registry.fill(config)
    assert not filled.is_interpolated
    assert filled["c"]["d"] == "${b}"
    resolved = my_registry.resolve(filled)
    assert resolved["c"]["d"] == "scratch!"


def test_config_deep_merge():
    config = {"a": "hello", "b": {"c": "d"}}
    defaults = {"a": "world", "b": {"c": "e", "f": "g"}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 2
    assert merged["a"] == "hello"
    assert merged["b"] == {"c": "d", "f": "g"}
    config = {"a": "hello", "b": {"@test": "x", "foo": 1}}
    defaults = {"a": "world", "b": {"@test": "x", "foo": 100, "bar": 2}, "c": 100}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@test": "x", "foo": 1, "bar": 2}
    assert merged["c"] == 100
    config = {"a": "hello", "b": {"@test": "x", "foo": 1}, "c": 100}
    defaults = {"a": "world", "b": {"@test": "y", "foo": 100, "bar": 2}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@test": "x", "foo": 1}
    assert merged["c"] == 100
    # Test that leaving out the factory just adds to existing
    config = {"a": "hello", "b": {"foo": 1}, "c": 100}
    defaults = {"a": "world", "b": {"@test": "y", "foo": 100, "bar": 2}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@test": "y", "foo": 1, "bar": 2}
    assert merged["c"] == 100
    # Test that switching to a different factory prevents the default from being added
    config = {"a": "hello", "b": {"@foo": 1}, "c": 100}
    defaults = {"a": "world", "b": {"@bar": "y"}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@foo": 1}
    assert merged["c"] == 100
    config = {"a": "hello", "b": {"@foo": 1}, "c": 100}
    defaults = {"a": "world", "b": "y"}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@foo": 1}
    assert merged["c"] == 100


def test_config_deep_merge_variables():
    config_str = """[a]\nb= 1\nc = 2\n\n[d]\ne = ${a:b}"""
    defaults_str = """[a]\nx = 100\n\n[d]\ny = 500"""
    config = Config().from_str(config_str, interpolate=False)
    defaults = Config().from_str(defaults_str)
    merged = defaults.merge(config)
    assert merged["a"] == {"b": 1, "c": 2, "x": 100}
    assert merged["d"] == {"e": "${a:b}", "y": 500}
    assert merged.interpolate()["d"] == {"e": 1, "y": 500}
    # With variable in defaults: overwritten by new value
    config = Config().from_str("""[a]\nb= 1\nc = 2""")
    defaults = Config().from_str("""[a]\nb = 100\nc = ${a:b}""", interpolate=False)
    merged = defaults.merge(config)
    assert merged["a"]["c"] == 2


def test_config_to_str_roundtrip():
    cfg = {"cfg": {"foo": False}}
    config_str = Config(cfg).to_str()
    assert config_str == "[cfg]\nfoo = false"
    config = Config().from_str(config_str)
    assert dict(config) == cfg
    cfg = {"cfg": {"foo": "false"}}
    config_str = Config(cfg).to_str()
    assert config_str == '[cfg]\nfoo = "false"'
    config = Config().from_str(config_str)
    assert dict(config) == cfg
    # Bad non-serializable value
    cfg = {"cfg": {"x": numpy.asarray([[1, 2, 3, 4], [4, 5, 3, 4]], dtype="f")}}
    config = Config(cfg)
    with pytest.raises(ConfigValidationError):
        config.to_str()
    # Roundtrip with variables: preserve variables correctly (quoted/unquoted)
    config_str = """[a]\nb = 1\n\n[c]\nd = ${a:b}\ne = \"hello${a:b}"\nf = "${a:b}\""""
    config = Config().from_str(config_str, interpolate=False)
    assert config.to_str() == config_str


def test_config_is_interpolated():
    """Test that a config object correctly reports whether it's interpolated."""
    config_str = """[a]\nb = 1\n\n[c]\nd = ${a:b}\ne = \"hello${a:b}"\nf = ${a}"""
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    config = config.merge(Config({"x": {"y": "z"}}))
    assert not config.is_interpolated
    config = Config(config)
    assert not config.is_interpolated
    config = config.interpolate()
    assert config.is_interpolated
    config = config.merge(Config().from_str(config_str, interpolate=False))
    assert not config.is_interpolated


@pytest.mark.parametrize(
    "section_order,expected_str,expected_keys",
    [
        # fmt: off
        ([], "[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4\n\n[h]\ni = 5\n\n[j]\nk = 6", ["a", "h", "j"]),
        (["j", "h", "a"], "[j]\nk = 6\n\n[h]\ni = 5\n\n[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4", ["j", "h", "a"]),
        (["h"], "[h]\ni = 5\n\n[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4\n\n[j]\nk = 6", ["h", "a", "j"])
        # fmt: on
    ],
)
def test_config_serialize_custom_sort(section_order, expected_str, expected_keys):
    cfg = {
        "j": {"k": 6},
        "a": {"b": 1, "d": {"e": 3}, "c": 2, "f": {"g": 4}},
        "h": {"i": 5},
    }
    cfg_str = Config(cfg).to_str()
    assert Config(cfg, section_order=section_order).to_str() == expected_str
    keys = list(Config(section_order=section_order).from_str(cfg_str).keys())
    assert keys == expected_keys
    keys = list(Config(cfg, section_order=section_order).keys())
    assert keys == expected_keys


def test_config_custom_sort_preserve():
    """Test that sort order is preserved when merging and copying configs,
    or when configs are filled and resolved."""
    cfg = {"x": {}, "y": {}, "z": {}}
    section_order = ["y", "z", "x"]
    expected = "[y]\n\n[z]\n\n[x]"
    config = Config(cfg, section_order=section_order)
    assert config.to_str() == expected
    config2 = config.copy()
    assert config2.to_str() == expected
    config3 = config.merge({"a": {}})
    assert config3.to_str() == f"{expected}\n\n[a]"
    config4 = Config(config)
    assert config4.to_str() == expected
    config_str = """[a]\nb = 1\n[c]\n@cats = "catsie.v1"\nevil = true\n\n[t]\n x = 2"""
    section_order = ["c", "a", "t"]
    config5 = Config(section_order=section_order).from_str(config_str)
    assert list(config5.keys()) == section_order
    filled = my_registry.fill(config5)
    assert filled.section_order == section_order


def test_config_pickle():
    config = Config({"foo": "bar"}, section_order=["foo", "bar", "baz"])
    data = pickle.dumps(config)
    config_new = pickle.loads(data)
    assert config_new == {"foo": "bar"}
    assert config_new.section_order == ["foo", "bar", "baz"]


def test_config_fill_extra_fields():
    """Test that filling a config from a schema removes extra fields."""

    class TestSchemaContent(BaseModel):
        a: str
        b: int

        class Config:
            extra = "forbid"

    class TestSchema(BaseModel):
        cfg: TestSchemaContent

    config = Config({"cfg": {"a": "1", "b": 2, "c": True}})
    with pytest.raises(ConfigValidationError):
        my_registry.fill(config, schema=TestSchema)
    filled = my_registry.fill(config, schema=TestSchema, validate=False)["cfg"]
    assert filled == {"a": "1", "b": 2}
    config2 = config.interpolate()
    filled = my_registry.fill(config2, schema=TestSchema, validate=False)["cfg"]
    assert filled == {"a": "1", "b": 2}
    config3 = Config({"cfg": {"a": "1", "b": 2, "c": True}}, is_interpolated=False)
    filled = my_registry.fill(config3, schema=TestSchema, validate=False)["cfg"]
    assert filled == {"a": "1", "b": 2}

    class TestSchemaContent2(BaseModel):
        a: str
        b: int

        class Config:
            extra = "allow"

    class TestSchema2(BaseModel):
        cfg: TestSchemaContent2

    filled = my_registry.fill(config, schema=TestSchema2, validate=False)["cfg"]
    assert filled == {"a": "1", "b": 2, "c": True}


def test_config_validation_error_custom():
    class Schema(BaseModel):
        hello: int
        world: int

    config = {"hello": 1, "world": "hi!"}
    with pytest.raises(ConfigValidationError) as exc_info:
        my_registry._fill(config, Schema)
    e1 = exc_info.value
    assert e1.title == "Config validation error"
    assert e1.desc is None
    assert not e1.parent
    assert e1.show_config is True
    assert len(e1.errors) == 1
    assert e1.errors[0]["loc"] == ("world",)
    assert e1.errors[0]["msg"] == "value is not a valid integer"
    assert e1.errors[0]["type"] == "type_error.integer"
    assert e1.error_types == set(["type_error.integer"])
    # Create a new error with overrides
    title = "Custom error"
    desc = "Some error description here"
    e2 = ConfigValidationError.from_error(e1, title=title, desc=desc, show_config=False)
    assert e2.errors == e1.errors
    assert e2.error_types == e1.error_types
    assert e2.title == title
    assert e2.desc == desc
    assert e2.show_config is False
    assert e1.text != e2.text


def test_config_parsing_error():
    config_str = "[a]\nb c"
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)


def test_config_fill_without_resolve():
    class BaseSchema(BaseModel):
        catsie: int

    config = {"catsie": {"@cats": "catsie.v1", "evil": False}}
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved["catsie"] == "meow"
    assert filled["catsie"]["cute"] is True
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(config, schema=BaseSchema)
    filled2 = my_registry.fill(config, schema=BaseSchema)
    assert filled2["catsie"]["cute"] is True
    resolved = my_registry.resolve(filled2)
    assert resolved["catsie"] == "meow"
    # With unavailable function
    class BaseSchema2(BaseModel):
        catsie: Any
        other: int = 12

    config = {"catsie": {"@cats": "dog", "evil": False}}
    filled3 = my_registry.fill(config, schema=BaseSchema2)
    assert filled3["catsie"] == config["catsie"]
    assert filled3["other"] == 12


def test_config_dataclasses():
    @my_registry.cats("catsie.ragged")
    def catsie_ragged(arg: Ragged):
        return arg

    data = numpy.zeros((20, 4), dtype="f")
    lengths = numpy.array([4, 2, 8, 1, 4], dtype="i")
    ragged = Ragged(data, lengths)
    config = {"cfg": {"@cats": "catsie.ragged", "arg": ragged}}
    result = my_registry.resolve(config)["cfg"]
    assert isinstance(result, Ragged)
    assert list(result._get_starts_ends()) == [0, 4, 6, 14, 15, 19]


@pytest.mark.parametrize(
    "greeting,value,expected",
    [
        # simple substitution should go fine
        [342, "${vars.a}", int],
        ["342", "${vars.a}", str],
        ["everyone", "${vars.a}", str],
    ],
)
def test_config_interpolates(greeting, value, expected):
    str_cfg = f"""
    [project]
    my_par = {value}

    [vars]
    a = "something"
    """
    overrides = {"vars.a": greeting}
    cfg = Config().from_str(str_cfg, overrides=overrides)
    assert type(cfg["project"]["my_par"]) == expected


@pytest.mark.parametrize(
    "greeting,value,expected",
    [
        # fmt: off
        # simple substitution should go fine
        ["hello 342", "${vars.a}", "hello 342"],
        ["hello everyone", "${vars.a}", "hello everyone"],
        ["hello tout le monde", "${vars.a}", "hello tout le monde"],
        ["hello 42", "${vars.a}", "hello 42"],
        # substituting an element in a list
        ["hello 342", "[1, ${vars.a}, 3]", "hello 342"],
        ["hello everyone", "[1, ${vars.a}, 3]", "hello everyone"],
        ["hello tout le monde", "[1, ${vars.a}, 3]", "hello tout le monde"],
        ["hello 42", "[1, ${vars.a}, 3]", "hello 42"],
        # substituting part of a string
        [342, "hello ${vars.a}", "hello 342"],
        ["everyone", "hello ${vars.a}", "hello everyone"],
        ["tout le monde", "hello ${vars.a}", "hello tout le monde"],
        pytest.param("42", "hello ${vars.a}", "hello 42", marks=pytest.mark.xfail),
        # substituting part of a implicit string inside a list
        [342, "[1, hello ${vars.a}, 3]", "hello 342"],
        ["everyone", "[1, hello ${vars.a}, 3]", "hello everyone"],
        ["tout le monde", "[1, hello ${vars.a}, 3]", "hello tout le monde"],
        pytest.param("42", "[1, hello ${vars.a}, 3]", "hello 42", marks=pytest.mark.xfail),
        # substituting part of a explicit string inside a list
        [342, "[1, 'hello ${vars.a}', '3']", "hello 342"],
        ["everyone", "[1, 'hello ${vars.a}', '3']", "hello everyone"],
        ["tout le monde", "[1, 'hello ${vars.a}', '3']", "hello tout le monde"],
        pytest.param("42", "[1, 'hello ${vars.a}', '3']", "hello 42", marks=pytest.mark.xfail),
        # more complicated example
        [342, "[{'name':'x','script':['hello ${vars.a}']}]", "hello 342"],
        ["everyone", "[{'name':'x','script':['hello ${vars.a}']}]", "hello everyone"],
        ["tout le monde", "[{'name':'x','script':['hello ${vars.a}']}]", "hello tout le monde"],
        pytest.param("42", "[{'name':'x','script':['hello ${vars.a}']}]", "hello 42", marks=pytest.mark.xfail),
        # fmt: on
    ],
)
def test_config_overrides(greeting, value, expected):
    str_cfg = f"""
    [project]
    commands = {value}

    [vars]
    a = "world"
    """
    overrides = {"vars.a": greeting}
    assert "${vars.a}" in str_cfg
    cfg = Config().from_str(str_cfg, overrides=overrides)
    assert expected in str(cfg)


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
