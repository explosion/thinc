from typing import Union, Dict, Any, Optional, List, Tuple, Callable, Type, Sequence
from types import GeneratorType
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from pydantic import BaseModel, create_model, ValidationError
from pydantic.main import ModelMetaclass
from wasabi import table
import srsly
import catalogue
import inspect
import io
import numpy

from .types import Decorator


def get_configparser():
    config = ConfigParser(interpolation=ExtendedInterpolation())
    # Preserve case of keys: https://stackoverflow.com/a/1611877/6400719
    config.optionxform = str
    return config


class Config(dict):
    """This class holds the model and training configuration and can load and
    save the TOML-style configuration format from/to a string, file or bytes.
    The Config class is a subclass of dict and uses Python's ConfigParser
    under the hood.
    """

    def __init__(
        self, data: Optional[Union[Dict[str, Any], "ConfigParser", "Config"]] = None
    ) -> None:
        """Initialize a new Config object with optional data."""
        dict.__init__(self)
        if data is None:
            data = {}
        self.update(data)

    def interpret_config(self, config: Union[Dict[str, Any], "ConfigParser"]):
        """Interpret a config, parse nested sections and parse the values
        as JSON. Mostly used internally and modifies the config in place.
        """
        for section, values in config.items():
            if section == "DEFAULT":
                # Skip [DEFAULT] section for now since it causes validation
                # errors and we don't want to use it
                continue
            parts = section.split(".")
            node = self
            for part in parts:
                node = node.setdefault(part, {})
            if not isinstance(node, dict):
                # Happens if both value *and* subsection were defined for a key
                err = [{"loc": parts, "msg": "found conflicting values"}]
                raise ConfigValidationError(f"{self}\n{({part: dict(values)})}", err)
            for key, value in values.items():
                node[key] = srsly.json_loads(config.get(section, key))

    def from_str(self, text: str) -> "Config":
        "Load the config from a string."
        config = get_configparser()
        config.read_string(text)
        for key in list(self.keys()):
            self.pop(key)
        self.interpret_config(config)
        return self

    def to_str(self) -> str:
        """Write the config to a string."""
        flattened = get_configparser()
        queue: List[Tuple[tuple, "Config"]] = [(tuple(), self)]
        for path, node in queue:
            for key, value in node.items():
                if hasattr(value, "items"):
                    queue.append((path + (key,), value))
                else:
                    assert path
                    section_name = ".".join(path)
                    if not flattened.has_section(section_name):
                        flattened.add_section(section_name)
                    flattened.set(section_name, key, srsly.json_dumps(value))
        string_io = io.StringIO()
        flattened.write(string_io)
        return string_io.getvalue().strip()

    def to_bytes(self) -> bytes:
        """Serialize the config to a byte string."""
        return self.to_str().encode("utf8")

    def from_bytes(self, bytes_data: bytes) -> "Config":
        """Load the config from a byte string."""
        return self.from_str(bytes_data.decode("utf8"))

    def to_disk(self, path: Union[str, Path]):
        """Serialize the config to a file."""
        path = Path(path)
        with path.open("w", encoding="utf8") as file_:
            file_.write(self.to_str())

    def from_disk(self, path: Union[str, Path]) -> "Config":
        """Load config from a file."""
        with Path(path).open("r", encoding="utf8") as file_:
            text = file_.read()
        return self.from_str(text)


class ConfigValidationError(ValueError):
    def __init__(
        self,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        errors: List[Dict[str, Any]],
        message: str = "Config validation error",
        element: str = "",
    ) -> None:
        """Custom error for validating configs."""
        data = []
        for error in errors:
            err_loc = " -> ".join([str(p) for p in error.get("loc", [])])
            if element:
                err_loc = f"{element} -> {err_loc}"
            data.append((err_loc, error.get("msg")))
        result = [message, table(data), f"{config}"]
        ValueError.__init__(self, "\n\n" + "\n".join(result))


ARGS_FIELD = "*"
ARGS_FIELD_ALIAS = "VARIABLE_POSITIONAL_ARGS"  # user is unlikely going to use this


class EmptySchema(BaseModel):
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


class _PromiseSchemaConfig:
    extra = "forbid"
    arbitrary_types_allowed = True
    # Underscore fields are not allowed in model, so use alias
    fields = {ARGS_FIELD_ALIAS: {"alias": ARGS_FIELD}}


class registry(object):
    # fmt: off
    optimizers: Decorator = catalogue.create("thinc", "optimizers", entry_points=True)
    schedules: Decorator = catalogue.create("thinc", "schedules", entry_points=True)
    layers: Decorator = catalogue.create("thinc", "layers", entry_points=True)
    losses: Decorator = catalogue.create("thinc", "losses", entry_points=True)
    initializers: Decorator = catalogue.create("thinc", "initializers", entry_points=True)
    datasets: Decorator = catalogue.create("thinc", "datasets", entry_points=True)
    # fmt: on

    @classmethod
    def create(cls, registry_name: str, entry_points: bool = False) -> None:
        """Create a new custom registry."""
        if hasattr(cls, registry_name):
            raise ValueError(f"Registry '{registry_name}' already exists")
        reg: Decorator = catalogue.create(
            "thinc", registry_name, entry_points=entry_points
        )
        setattr(cls, registry_name, reg)

    @classmethod
    def get(cls, registry_name: str, func_name: str) -> Callable:
        """Get a registered function from a given registry."""
        if not hasattr(cls, registry_name):
            raise ValueError(f"Unknown registry: '{registry_name}'")
        reg = getattr(cls, registry_name)
        func = reg.get(func_name)
        if func is None:
            raise ValueError(f"Could not find '{func_name}' in '{registry_name}'")
        return func

    @classmethod
    def make_from_config(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema: Type[BaseModel] = EmptySchema,
        validate: bool = True,
    ) -> Config:
        """Unpack a config dictionary, creating objects from the registry
        recursively. If validate=True, the config will be validated against the
        type annotations of the registered functions referenced in the config
        (if available) and/or the schema (if available).
        """
        # Valid: {"optimizer": {"@optimizers": "my_cool_optimizer", "rate": 1.0}}
        # Invalid: {"@optimizers": "my_cool_optimizer", "rate": 1.0}
        if cls.is_promise(config):
            err_msg = "The top-level config object can't be a reference to a registered function."
            raise ConfigValidationError(config, [{"msg": err_msg}])
        _, _, resolved = cls._fill(config, schema, validate)
        return resolved

    @classmethod
    def fill_config(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema: Type[BaseModel] = EmptySchema,
        validate: bool = True,
    ) -> Config:
        """Unpack a config dictionary, leave all references to registry
        functions intact and don't resolve them, but fill in all values and
        defaults based on the type annotations. If validate=True, the config
        will be validated against the type annotations of the registered
        functions referenced in the config (if available) and/or the schema
        (if available).
        """
        # Valid: {"optimizer": {"@optimizers": "my_cool_optimizer", "rate": 1.0}}
        # Invalid: {"@optimizers": "my_cool_optimizer", "rate": 1.0}
        if cls.is_promise(config):
            err_msg = "The top-level config object can't be a reference to a registered function."
            raise ConfigValidationError(config, [{"msg": err_msg}])
        filled, _, _ = cls._fill(config, schema, validate)
        return filled

    @classmethod
    def _fill(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        schema: Type[BaseModel] = EmptySchema,
        validate: bool = True,
        parent: str = "",
    ) -> Tuple[Config, Config, Config]:
        """Build three representations of the config:
        1. All promises are preserved (just like config user would provide).
        2. Promises are replaced by their return values. This is the validation
           copy and will be parsed by pydantic. It lets us include hacks to
           work around problems (e.g. handling of generators).
        3. Final copy with promises replaced by their return values. This is
           what registry.make_from_config returns.
        """
        filled: Dict[str, Any] = {}
        validation: Dict[str, Any] = {}
        final: Dict[str, Any] = {}
        for key, value in config.items():
            key_parent = f"{parent}.{key}".strip(".")
            if cls.is_promise(value):
                promise_schema = cls.make_promise_schema(value)
                filled[key], validation[key], final[key] = cls._fill(
                    value, promise_schema, validate, parent=key_parent
                )
                # Call the function and populate the field value. We can't just
                # create an instance of the type here, since this wouldn't work
                # for generics / more complex custom types
                getter = cls.get_constructor(final[key])
                args, kwargs = cls.parse_args(final[key])
                try:
                    getter_result = getter(*args, **kwargs)
                except Exception as err:
                    err_msg = "Can't construct config: calling registry function failed"
                    raise ConfigValidationError(
                        {key: value}, [{"msg": err, "loc": [getter.__name__]}], err_msg
                    )
                validation[key] = getter_result
                final[key] = getter_result
                if isinstance(validation[key], GeneratorType):
                    # If value is a generator we can't validate type without
                    # consuming it (which doesn't work if it's infinite – see
                    # schedule for examples). So we skip it.
                    validation[key] = []
            elif hasattr(value, "items"):
                field_type = EmptySchema
                if key in schema.__fields__:
                    field = schema.__fields__[key]
                    field_type = field.type_
                    if not isinstance(field.type_, ModelMetaclass):
                        # If we don't have a pydantic schema and just a type
                        field_type = EmptySchema
                filled[key], validation[key], final[key] = cls._fill(
                    value, field_type, validate, parent=key_parent
                )
                if key == ARGS_FIELD and isinstance(validation[key], dict):
                    # If the value of variable positional args is a dict (e.g.
                    # created via config blocks), only use its values
                    validation[key] = list(validation[key].values())
                    final[key] = list(final[key].values())
            else:
                filled[key] = value
                # Prevent pydantic from consuming generator if part of a union
                validation[key] = value if not isinstance(value, GeneratorType) else []
                final[key] = value
        # Now that we've filled in all of the promises, update with defaults
        # from schema, and validate if validation is enabled
        if validate:
            try:
                result = schema.parse_obj(validation)
            except ValidationError as e:
                raise ConfigValidationError(config, e.errors(), element=parent)
        else:
            # Same as parse_obj, but without validation
            result = schema.construct(**validation)
        validation.update(result.dict(exclude={ARGS_FIELD_ALIAS}))
        filled, final = cls._update_from_parsed(validation, filled, final)
        return Config(filled), Config(validation), Config(final)

    @classmethod
    def _update_from_parsed(
        cls, validation: Dict[str, Any], filled: Dict[str, Any], final: Dict[str, Any]
    ):
        """Update the final result with the parsed config like converted
        values recursively.
        """
        for key, value in validation.items():
            if key not in filled:
                filled[key] = value
            if key not in final:
                final[key] = value
            if isinstance(value, dict):
                filled[key], final[key] = cls._update_from_parsed(
                    value, filled[key], final[key]
                )
            # Update final config with parsed value if they're not equal (in
            # value and in type) but not if it's a generator because we had to
            # replace that to validate it correctly
            elif key == ARGS_FIELD:
                continue  # don't substitute if list of positional args
            elif isinstance(value, numpy.ndarray):  # check numpy first, just in case
                final[key] = value
            elif (
                value != final[key] or not isinstance(type(value), type(final[key]))
            ) and not isinstance(final[key], GeneratorType):
                final[key] = value
        return filled, final

    @classmethod
    def is_promise(cls, obj: Any) -> bool:
        """Check whether an object is a "promise", i.e. contains a reference
        to a registered function (via a key starting with `"@"`.
        """
        if not hasattr(obj, "keys"):
            return False
        id_keys = [k for k in obj.keys() if k.startswith("@")]
        if len(id_keys):
            return True
        return False

    @classmethod
    def get_constructor(cls, obj: Dict[str, Any]) -> Callable:
        id_keys = [k for k in obj.keys() if k.startswith("@")]
        if len(id_keys) != 1:
            err_msg = f"A block can only contain one function registry reference. Got: {id_keys}"
            raise ConfigValidationError(obj, [{"msg": err_msg}])
        else:
            key = id_keys[0]
            value = obj[key]
            return cls.get(key[1:], value)

    @classmethod
    def parse_args(cls, obj: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        args = []
        kwargs = {}
        for key, value in obj.items():
            if not key.startswith("@"):
                if key == ARGS_FIELD:
                    args = value
                else:
                    kwargs[key] = value
        return args, kwargs

    @classmethod
    def make_promise_schema(cls, obj: Dict[str, Any]) -> Type[BaseModel]:
        """Create a schema for a promise dict (referencing a registry function)
        by inspecting the function signature.
        """
        func = cls.get_constructor(obj)
        # Read the argument annotations and defaults from the function signature
        id_keys = [k for k in obj.keys() if k.startswith("@")]
        sig_args: Dict[str, Any] = {id_keys[0]: (str, ...)}
        for param in inspect.signature(func).parameters.values():
            # If no annotation is specified assume it's anything
            annotation = param.annotation if param.annotation != param.empty else Any
            # If no default value is specified assume that it's required
            default = param.default if param.default != param.empty else ...
            # Handle spread arguments and use their annotation as Sequence[whatever]
            if param.kind == param.VAR_POSITIONAL:
                spread_annot = Sequence[annotation]  # type: ignore
                sig_args[ARGS_FIELD_ALIAS] = (spread_annot, default)
            else:
                sig_args[param.name] = (annotation, default)
        sig_args["__config__"] = _PromiseSchemaConfig
        return create_model("ArgModel", **sig_args)


__all__ = ["Config", "registry", "ConfigValidationError"]
