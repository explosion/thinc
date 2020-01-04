import pytest
from typing import Iterable, Union, Sequence
from pydantic import BaseModel, StrictBool, StrictFloat, PositiveInt, constr
import catalogue
import thinc._registry
from thinc._registry import ConfigValidationError
from thinc.types import Generator
from thinc.config import Config
from thinc.optimizers import Adam  # noqa: F401
from thinc.schedules import warmup_linear  # noqa: F401


class my_registry(thinc._registry.registry):
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

EXAMPLE_CONFIG = """
[DEFAULT]

[optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
use_averages = true

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
start = 0.1
steps = 10000

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
@layers = "thinc.Affine.v1"
"""

OPTIMIZER_CFG = """
[optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
use_averages = true

[optimizer.schedules.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.1
warmup_steps = 10000
total_steps = 100000
"""


def test_validate_simple_config():
    simple_config = {"hello": 1, "world": 2}
    f, v = my_registry._fill(simple_config, HelloIntsSchema)
    assert f == simple_config
    assert v == simple_config


def test_invalidate_simple_config():
    invalid_config = {"hello": 1, "world": "hi!"}
    with pytest.raises(ConfigValidationError):
        f, v = my_registry._fill(invalid_config, HelloIntsSchema)


def test_invalidate_extra_args():
    invalid_config = {"hello": 1, "world": 2, "extra": 3}
    with pytest.raises(ConfigValidationError):
        f, v = my_registry._fill(invalid_config, HelloIntsSchema)


def test_fill_defaults_simple_config():
    valid_config = {"required": 1}
    filled, v = my_registry._fill(valid_config, DefaultsSchema)
    assert filled["required"] == 1
    assert filled["optional"] == "default value"
    invalid_config = {"optional": "some value"}
    with pytest.raises(ConfigValidationError):
        f, v = my_registry._fill(invalid_config, DefaultsSchema)


def test_fill_recursive_config():
    valid_config = {"outer_req": 1, "level2_req": {"hello": 4, "world": 7}}
    filled, validation = my_registry._fill(valid_config, ComplexSchema)
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
    filled, validated = my_registry._fill(config, DefaultsSchema)
    assert filled == config
    assert validated == {"required": 1, "optional": "meow"}


def test_fill_validate_promise():
    config = {"required": 1, "optional": {"@cats": "catsie.v1", "evil": False}}
    filled, validated = my_registry._fill(config, DefaultsSchema)
    assert filled["optional"]["cute"] is True


def test_fill_invalidate_promise():
    config = {"required": 1, "optional": {"@cats": "catsie.v1", "evil": False}}
    with pytest.raises(ConfigValidationError):
        filled, validated = my_registry._fill(config, HelloIntsSchema)
    config["optional"]["whiskers"] = True
    with pytest.raises(ConfigValidationError):
        filled, validated = my_registry._fill(config, DefaultsSchema)


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


def test_read_config():
    byte_string = EXAMPLE_CONFIG.encode("utf8")
    cfg = Config().from_bytes(byte_string)
    assert cfg["optimizer"]["learn_rate"]["start"] == 0.1
    assert cfg["pipeline"]["parser"]["factory"] == "parser"
    assert cfg["pipeline"]["parser"]["model"]["tok2vec"]["width"] == 128


def test_optimizer_config():
    cfg = Config().from_str(OPTIMIZER_CFG)
    result = my_registry.make_from_config(cfg)
    optimizer = result["optimizer"]
    assert optimizer.b1 == 0.9


def test_config_to_str():
    cfg = Config().from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()


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


def test_validation_no_validate():
    config = {"one": 1, "two": {"three": {"@cats": "catsie.v1", "evil": "false"}}}
    result = my_registry.make_from_config(config, validate=False)
    assert result["one"] == 1
    assert result["two"] == {"three": "scratch!"}


def test_validation_fill_defaults():
    config = {"one": 1, "two": {"@cats": "catsie.v1"}}
    result = my_registry.fill_config(config, validate=False)
    assert len(result["two"]) == 2  # no value filled in for "evil"
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


def test_validation_generators_iterable():
    @thinc.registry.optimizers("test_optimizer.v1")
    def test_optimizer_v1(
        rate: float, schedule: Union[float, Sequence[float], Generator]
    ) -> None:
        return None

    @thinc.registry.schedules("test_schedule.v1")
    def test_schedule_v1(some_value: float = 1.0) -> Iterable[float]:
        while True:
            yield some_value

    config = {
        "optimizer": {
            "@optimizers": "test_optimizer.v1",
            "rate": 0.1,
            "schedule": {"@schedules": "test_schedule.v1", "some_value": 1.0},
        }
    }
    my_registry.make_from_config(config)


def test_validation_unset_type_hints():
    """Test that unset type hints are handled correctly (and treated as Any)."""

    @thinc.registry.optimizers("test_optimizer.v2")
    def test_optimizer_v2(rate, steps: int = 10) -> None:
        return None

    config = {"test": {"@optimizers": "test_optimizer.v2", "rate": 0.1, "steps": 20}}
    my_registry.make_from_config(config)
