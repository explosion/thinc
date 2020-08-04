from typing import Union, Dict, Any, Optional, List, Tuple, Callable, Type, Sequence
from types import GeneratorType
from configparser import ConfigParser, ExtendedInterpolation, MAX_INTERPOLATION_DEPTH
from configparser import InterpolationMissingOptionError, InterpolationSyntaxError
from configparser import NoSectionError, NoOptionError, InterpolationDepthError
from pathlib import Path
from pydantic import BaseModel, create_model, ValidationError
from pydantic.main import ModelMetaclass
from wasabi import table
import srsly
import catalogue
import inspect
import io
import numpy
import copy

from .types import Decorator


SECTION_PREFIX = "__SECTION__:"


class CustomInterpolation(ExtendedInterpolation):
    def before_read(self, parser, section, option, value):
        # If we're dealing with a quoted string as the interpolation value,
        # make sure we load and unquote it so we don't end up with '"value"'
        try:
            json_value = srsly.json_loads(value)
            if isinstance(json_value, str):
                value = json_value
        except Exception:
            pass
        return super().before_read(parser, section, option, value)

    def before_get(self, parser, section, option, value, defaults):
        # Mostly copy-pasted from the built-in configparser implementation.
        L = []
        self.interpolate(parser, option, L, value, section, defaults, 1)
        return "".join(L)

    def interpolate(self, parser, option, accum, rest, section, map, depth):
        # Mostly copy-pasted from the built-in configparser implementation.
        # We need to overwrite this method so we can add special handling for
        # block references :( All values produced here should be strings –
        # we need to wait until the whole config is interpreted anyways so
        # filling in incomplete values here is pointless. All we need is the
        # section reference so we can fetch it later.
        rawval = parser.get(section, option, raw=True, fallback=rest)
        if depth > MAX_INTERPOLATION_DEPTH:
            raise InterpolationDepthError(option, section, rawval)
        while rest:
            p = rest.find("$")
            if p < 0:
                accum.append(rest)
                return
            if p > 0:
                accum.append(rest[:p])
                rest = rest[p:]
            # p is no longer used
            c = rest[1:2]
            if c == "$":
                accum.append("$")
                rest = rest[2:]
            elif c == "{":
                m = self._KEYCRE.match(rest)
                if m is None:
                    err = f"bad interpolation variable reference {rest}"
                    raise InterpolationSyntaxError(option, section, err)
                path = m.group(1).split(":")
                rest = rest[m.end() :]
                sect = section
                opt = option
                try:
                    if len(path) == 1:
                        opt = parser.optionxform(path[0])
                        if opt in map:
                            v = map[opt]
                        else:
                            # We have block reference, store it as a special key
                            section_name = parser[parser.optionxform(path[0])]._name
                            v = f"{SECTION_PREFIX}{section_name}"
                    elif len(path) == 2:
                        sect = path[0]
                        opt = parser.optionxform(path[1])
                        v = parser.get(sect, opt, raw=True)
                    else:
                        err = f"More than one ':' found: {rest}"
                        raise InterpolationSyntaxError(option, section, err)
                except (KeyError, NoSectionError, NoOptionError):
                    raise InterpolationMissingOptionError(
                        option, section, rawval, ":".join(path)
                    ) from None
                if "$" in v:
                    new_map = dict(parser.items(sect, raw=True))
                    self.interpolate(parser, opt, accum, v, sect, new_map, depth + 1)
                else:
                    accum.append(v)
            else:
                err = "'$' must be followed by '$' or '{', " "found: %r" % (rest,)
                raise InterpolationSyntaxError(option, section, err)


def get_configparser():
    config = ConfigParser(interpolation=CustomInterpolation())
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

    def interpret_config(self, config: "ConfigParser") -> None:
        """Interpret a config, parse nested sections and parse the values
        as JSON. Mostly used internally and modifies the config in place.
        """
        # Sort sections by depth, so that we can iterate breadth-first. This
        # allows us to check that we're not expanding an undefined block.
        get_depth = lambda item: len(item[0].split("."))
        for section, values in sorted(config.items(), key=get_depth):
            if section == "DEFAULT":
                # Skip [DEFAULT] section for now since it causes validation
                # errors and we don't want to use it
                continue
            parts = section.split(".")
            node = self
            for part in parts[:-1]:
                if part == "*":
                    node = node.setdefault(part, {})
                elif part not in node:
                    err_title = f"Error parsing config section. Perhaps a section name is wrong?"
                    err = [{"loc": parts, "msg": f"Section '{part}' is not defined"}]
                    raise ConfigValidationError(self, err, message=err_title)
                else:
                    node = node[part]
            # TODO: add error if node not in list
            node = node.setdefault(parts[-1], {})
            if not isinstance(node, dict):
                # Happens if both value *and* subsection were defined for a key
                err = [{"loc": parts, "msg": "found conflicting values"}]
                raise ConfigValidationError(f"{self}\n{({part: dict(values)})}", err)
            try:
                keys_values = list(values.items())
            except InterpolationMissingOptionError as e:
                err_msg = (
                    "If you're using variables referring to sub-sections, make "
                    "sure they're devided by a colon (:) not a dot. For example: "
                    "${section:subsection}"
                )
                raise ConfigValidationError(f"{e}\n\n{err_msg}", [])
            for key, value in keys_values:
                config_v = config.get(section, key)
                try:
                    node[key] = srsly.json_loads(config_v)
                except Exception:
                    node[key] = config_v
        self.replace_section_refs(self)

    def replace_section_refs(self, config: Union[Dict[str, Any], "Config"]) -> None:
        """Replace references to section blocks in the final config."""
        for key, value in config.items():
            if isinstance(value, dict):
                self.replace_section_refs(value)
            elif isinstance(value, str) and value.startswith(SECTION_PREFIX):
                parts = value.replace(SECTION_PREFIX, "").split(".")
                result = self
                for item in parts:
                    try:
                        result = result[item]
                    except (KeyError, TypeError):  # This should never happen
                        err_title = "Error parsing reference to config section"
                        err_msg = f"Section '{'.'.join(parts)}' is not defined"
                        err = [{"loc": parts, "msg": err_msg}]
                        raise ConfigValidationError(self, err, message=err_title)
                config[key] = result

    def copy(self) -> "Config":
        """Deepcopy the config."""
        try:
            config = copy.deepcopy(self)
        except Exception as e:
            raise ValueError(f"Couldn't deep-copy config: {e}")
        return Config(config)

    def _set_overrides(self, config: "ConfigParser", overrides: Dict[str, Any]) -> None:
        """Set overrides in the ConfigParser before config is interpreted."""
        err_title = "Error parsing config overrides"
        for key, value in overrides.items():
            err_msg = "not a section value that can be overwritten"
            err = [{"loc": key.split("."), "msg": err_msg}]
            if "." not in key:
                raise ConfigValidationError("", err, message=err_title)
            section, option = key.rsplit(".", 1)
            if section not in config or option not in config[section]:
                raise ConfigValidationError("", err, message=err_title)
            config.set(section, option, srsly.json_dumps(value))

    def from_str(self, text: str, *, overrides: Dict[str, Any] = {}) -> "Config":
        """Load the config from a string."""
        config = get_configparser()
        config.read_string(text)
        self._set_overrides(config, overrides)
        for key in list(self.keys()):
            self.pop(key)
        self.interpret_config(config)
        return self

    def to_str(self) -> str:
        """Write the config to a string."""
        flattened = get_configparser()
        queue: List[Tuple[tuple, "Config"]] = [(tuple(), self)]
        for path, node in queue:
            section_name = ".".join(path)
            if path and path[-1] != "*" and not flattened.has_section(section_name):
                # Always create sections for non-'*' sections, not only if
                # they have leaf entries, as we don't want to expand
                # blocks that are undefined
                flattened.add_section(section_name)
            for key, value in node.items():
                if hasattr(value, "items"):
                    # Reference to a function with no arguments, serialize
                    # inline as a dict and don't create new section
                    if registry.is_promise(value) and len(value) == 1:
                        flattened.set(section_name, key, srsly.json_dumps(value))
                    else:
                        queue.append((path + (key,), value))
                else:
                    flattened.set(section_name, key, srsly.json_dumps(value))
        # Order so subsection follow parent (not all sections, then all subs etc.)
        flattened._sections = dict(
            sorted(flattened._sections.items(), key=lambda x: x[0])
        )
        string_io = io.StringIO()
        flattened.write(string_io)
        return string_io.getvalue().strip()

    def to_bytes(self) -> bytes:
        """Serialize the config to a byte string."""
        return self.to_str().encode("utf8")

    def from_bytes(
        self, bytes_data: bytes, *, overrides: Dict[str, Any] = {}
    ) -> "Config":
        """Load the config from a byte string."""
        return self.from_str(bytes_data.decode("utf8"), overrides=overrides)

    def to_disk(self, path: Union[str, Path]):
        """Serialize the config to a file."""
        path = Path(path)
        with path.open("w", encoding="utf8") as file_:
            file_.write(self.to_str())

    def from_disk(
        self, path: Union[str, Path], *, overrides: Dict[str, Any] = {}
    ) -> "Config":
        """Load config from a file."""
        with Path(path).open("r", encoding="utf8") as file_:
            text = file_.read()
        return self.from_str(text, overrides=overrides)


class ConfigValidationError(ValueError):
    def __init__(
        self,
        config: Union[Config, Dict[str, Dict[str, Any]], str],
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
        data_table = table(data) if data else ""
        result = [message, data_table, f"{config}"]
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
    def resolve(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema: Type[BaseModel] = EmptySchema,
        overrides: Dict[str, Any] = {},
        validate: bool = True,
    ) -> Tuple[Config, Config]:
        """Unpack a config dictionary and create two versions of the config:
        a resolved version with objects from the registry created recursively,
        and a filled version with all references to registry functions left
        intact, but filled with all values and defaults based on the type
        annotations. If validate=True, the config will be validated against the
        type annotations of the registered functions referenced in the config
        (if available) and/or the schema (if available).
        """
        # Valid: {"optimizer": {"@optimizers": "my_cool_optimizer", "rate": 1.0}}
        # Invalid: {"@optimizers": "my_cool_optimizer", "rate": 1.0}
        if cls.is_promise(config):
            err_msg = "The top-level config object can't be a reference to a registered function."
            raise ConfigValidationError(config, [{"msg": err_msg}])
        filled, _, resolved = cls._fill(
            config, schema, validate=validate, overrides=overrides
        )
        # Check that overrides didn't include invalid properties not in config
        if validate:
            cls._validate_overrides(filled, overrides)
        return resolved, filled

    @classmethod
    def make_from_config(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema: Type[BaseModel] = EmptySchema,
        overrides: Dict[str, Any] = {},
        validate: bool = True,
    ) -> Config:
        """Unpack a config dictionary, creating objects from the registry
        recursively. If validate=True, the config will be validated against the
        type annotations of the registered functions referenced in the config
        (if available) and/or the schema (if available).
        """
        # Valid: {"optimizer": {"@optimizers": "my_cool_optimizer", "rate": 1.0}}
        # Invalid: {"@optimizers": "my_cool_optimizer", "rate": 1.0}
        resolved, _ = cls.resolve(
            config, schema=schema, overrides=overrides, validate=validate
        )
        return resolved

    @classmethod
    def fill_config(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema: Type[BaseModel] = EmptySchema,
        overrides: Dict[str, Any] = {},
        validate: bool = True,
    ) -> Config:
        """Unpack a config dictionary, leave all references to registry
        functions intact and don't resolve them, but fill in all values and
        defaults based on the type annotations. If validate=True, the config
        will be validated against the type annotations of the registered
        functions referenced in the config (if available) and/or the schema
        (if available).
        """
        _, filled = cls.resolve(
            config, schema=schema, overrides=overrides, validate=validate
        )
        return filled

    @classmethod
    def _fill(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        schema: Type[BaseModel] = EmptySchema,
        *,
        validate: bool = True,
        parent: str = "",
        overrides: Dict[str, Dict[str, Any]] = {},
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
            if key_parent in overrides:
                value = overrides[key_parent]
                config[key] = value
            if cls.is_promise(value):
                promise_schema = cls.make_promise_schema(value)
                filled[key], validation[key], final[key] = cls._fill(
                    value,
                    promise_schema,
                    validate=validate,
                    parent=key_parent,
                    overrides=overrides,
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
                if isinstance(validation[key], dict):
                    # The registered function returned a dict, prevent it from
                    # being validated as a config section
                    validation[key] = {}
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
                    value,
                    field_type,
                    validate=validate,
                    parent=key_parent,
                    overrides=overrides,
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
    def _validate_overrides(cls, filled: Config, overrides: Dict[str, Any]):
        """Validate overrides against a filled config to make sure there are
        no references to properties that don't exist and weren't used."""
        error_msg = "Invalid override: config value doesn't exist"
        errors = []
        for override_key in overrides.keys():
            if not cls._is_in_config(override_key, filled):
                errors.append({"msg": error_msg, "loc": [override_key]})
        if errors:
            raise ConfigValidationError(filled, errors)

    @classmethod
    def _is_in_config(cls, prop: str, config: Union[Dict[str, Any], Config]):
        """Check whether a nested config property like "section.subsection.key"
        is in a given config."""
        tree = prop.split(".")
        obj = dict(config)
        while tree:
            key = tree.pop(0)
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return False
        return True

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
