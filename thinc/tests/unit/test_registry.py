import pytest
from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError
import catalogue
import thinc._registry


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


def test_validate_simple_config():
    simple_config = {"hello": 1, "world": 2}
    f, v = my_registry.fill_and_validate(simple_config, HelloIntsSchema)
    assert f == simple_config
    assert v == simple_config


def test_invalidate_simple_config():
    invalid_config = {"hello": 1, "world": "hi!"}
    with pytest.raises(ValidationError):
        f, v = my_registry.fill_and_validate(invalid_config, HelloIntsSchema)


def test_invalidate_extra_args():
    invalid_config = {"hello": 1, "world": 2, "extra": 3}
    with pytest.raises(ValidationError):
        f, v = my_registry.fill_and_validate(invalid_config, HelloIntsSchema)


def test_fill_defaults_simple_config():
    valid_config = {"required": 1}
    filled, v = my_registry.fill_and_validate(valid_config, DefaultsSchema)
    assert filled["required"] == 1
    assert filled["optional"] == "default value"
    invalid_config = {"optional": "some value"}
    with pytest.raises(ValidationError):
        f, v = my_registry.fill_and_validate(invalid_config, DefaultsSchema)


def test_fill_recursive_config():
    valid_config = {"outer_req": 1, "level2_req": {"hello": 4, "world": 7}}
    filled, validation = my_registry.fill_and_validate(valid_config, ComplexSchema)
    assert filled["outer_req"] == 1
    assert filled["outer_opt"] == "default value"
    assert filled["level2_req"]["hello"] == 4
    assert filled["level2_req"]["world"] == 7
    assert filled["level2_opt"]["required"] == 1
    assert filled["level2_opt"]["optional"] == "default value"


@my_registry.cats.register("catsie.v1")
def catsie_v1(evil: bool, cute: bool = True) -> str:
    if evil:
        return "scratch!"
    else:
        return "meow"


good_catsie = {"@cats": "catsie.v1", "evil": False, "cute": True}

ok_catsie = {"@cats": "catsie.v1", "evil": False, "cute": False}
bad_catsie = {"@cats": "catsie.v1", "evil": True, "cute": True}

worst_catsie = {"@cats": "catsie.v1", "evil": True, "cute": False}


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
    filled, validated = my_registry.fill_and_validate(config, DefaultsSchema)
    assert filled == config
    assert validated == {"required": 1, "optional": ""}


def test_fill_validate_promise():
    config = {"required": 1, "optional": {"@cats": "catsie.v1", "evil": False}}
    filled, validated = my_registry.fill_and_validate(config, DefaultsSchema)
    assert filled["optional"]["cute"] is True


def test_fill_invalidate_promise():
    config = {"required": 1, "optional": {"@cats": "catsie.v1", "evil": False}}

    with pytest.raises(ValidationError):
        filled, validated = my_registry.fill_and_validate(config, HelloIntsSchema)
    config["optional"]["whiskers"] = True
    with pytest.raises(ValidationError):
        filled, validated = my_registry.fill_and_validate(config, DefaultsSchema)
