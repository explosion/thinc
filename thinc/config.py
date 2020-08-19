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
import re

from .types import Decorator


# Field used for positional arguments, e.g. [section.*.xyz]. The alias is
# required for the schema (shouldn't clash with user-defined arg names)
ARGS_FIELD = "*"
ARGS_FIELD_ALIAS = "VARIABLE_POSITIONAL_ARGS"
# Aliases for fields that would otherwise shadow pydantic attributes. Can be any
# string, so we're using name + space so it looks the same in error messages etc.
RESERVED_FIELDS = {"validate": "validate\u0020"}
# Internal prefix used to mark section references for custom interpolation
SECTION_PREFIX = "__SECTION__:"
# Values that shouldn't be loaded during interpolation because it'd cause
# even explicit string values to be incorrectly parsed as bools/None etc.
JSON_EXCEPTIONS = ("true", "false", "null")
# Regex to detect whether a value contains a variable
VARIABLE_RE = re.compile(r"\$\{[\w\.:]+\}")


class CustomInterpolation(ExtendedInterpolation):
    def before_read(self, parser, section, option, value):
        # If we're dealing with a quoted string as the interpolation value,
        # make sure we load and unquote it so we don't end up with '"value"'
        try:
            json_value = srsly.json_loads(value)
            if isinstance(json_value, str) and json_value not in JSON_EXCEPTIONS:
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
                # We want to treat both ${a:b} and ${a.b} the same
                rest = rest.replace(":", ".")
                m = self._KEYCRE.match(rest)
                if m is None:
                    err = f"bad interpolation variable reference {rest}"
                    raise InterpolationSyntaxError(option, section, err)
                path = m.group(1).rsplit(".", 1)
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
                        fallback = "__FALLBACK__"
                        v = parser.get(sect, opt, raw=True, fallback=fallback)
                        # If a variable doesn't exist, try again and treat the
                        # reference as a section
                        if v == fallback:
                            v = f"{SECTION_PREFIX}{parser[f'{sect}.{opt}']._name}"
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


def get_configparser(interpolate: bool = True):
    config = ConfigParser(interpolation=CustomInterpolation() if interpolate else None)
    # Preserve case of keys: https://stackoverflow.com/a/1611877/6400719
    config.optionxform = str  # type: ignore
    return config


class Config(dict):
    """This class holds the model and training configuration and can load and
    save the TOML-style configuration format from/to a string, file or bytes.
    The Config class is a subclass of dict and uses Python's ConfigParser
    under the hood.
    """

    is_interpolated: bool

    def __init__(
        self,
        data: Optional[Union[Dict[str, Any], "ConfigParser", "Config"]] = None,
        *,
        is_interpolated: Optional[bool] = None,
        section_order: Optional[List[str]] = None,
    ) -> None:
        """Initialize a new Config object with optional data."""
        dict.__init__(self)
        if data is None:
            data = {}
        if not isinstance(data, (dict, Config, ConfigParser)):
            raise ValueError(
                f"Can't initialize Config with data. Expected dict, Config or "
                f"ConfigParser but got: {type(data)}"
            )
        # Whether the config has been interpolated. We can use this to check
        # whether we need to interpolate again when it's resolved. We assume
        # that a config is interpolated by default.
        if is_interpolated is not None:
            self.is_interpolated = is_interpolated
        elif isinstance(data, Config):
            self.is_interpolated = data.is_interpolated
        else:
            self.is_interpolated = True
        if section_order is not None:
            self.section_order = section_order
        elif isinstance(data, Config):
            self.section_order = data.section_order
        else:
            self.section_order = []
        # Update with data
        self.update(self._sort(data))

    def interpolate(self) -> "Config":
        """Interpolate a config. Returns a copy of the object."""
        # This is currently the most effective way because we need our custom
        # to_str logic to run in order to re-serialize the values so we can
        # interpolate them again. ConfigParser.read_dict will just call str()
        # on all values, which isn't enough.
        return Config().from_str(self.to_str())

    def interpret_config(self, config: "ConfigParser") -> None:
        """Interpret a config, parse nested sections and parse the values
        as JSON. Mostly used internally and modifies the config in place.
        """
        self._validate_sections(config)
        # Sort sections by depth, so that we can iterate breadth-first. This
        # allows us to check that we're not expanding an undefined block.
        get_depth = lambda item: len(item[0].split("."))
        for section, values in sorted(config.items(), key=get_depth):
            if section == "DEFAULT":
                # Skip [DEFAULT] section so it doesn't cause validation error
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
                raise ConfigValidationError(f"{e}", []) from None
            for key, value in keys_values:
                config_v = config.get(section, key)
                if VARIABLE_RE.search(config_v):
                    node[key] = config_v
                else:
                    try:
                        node[key] = srsly.json_loads(config_v)
                    except Exception:
                        node[key] = config_v
        self.replace_section_refs(self)

    def replace_section_refs(
        self, config: Union[Dict[str, Any], "Config"], parent: str = ""
    ) -> None:
        """Replace references to section blocks in the final config."""
        for key, value in config.items():
            key_parent = f"{parent}.{key}".strip(".")
            if isinstance(value, dict):
                self.replace_section_refs(value, parent=key_parent)
            elif isinstance(value, str) and value.startswith(SECTION_PREFIX):
                parts = value.replace(SECTION_PREFIX, "").split(".")
                result = self
                for item in parts:
                    try:
                        result = result[item]
                    except (KeyError, TypeError):  # This should never happen
                        err_title = "Error parsing reference to config section"
                        err_msg = f"Section '{'.'.join(parts)}' is not defined"
                        raise ConfigValidationError(
                            self, [{"loc": parts, "msg": err_msg}], message=err_title
                        ) from None
                config[key] = result
            elif isinstance(value, str) and SECTION_PREFIX in value:
                # String value references a section (either a dict or return
                # value of promise). We can't allow this, since variables are
                # always interpolated *before* configs are resolved.
                err_title = (
                    "Can't reference whole sections or return values of function "
                    "blocks inside a string\n\nYou can change your variable to "
                    "reference a value instead. Keep in mind that it's not "
                    "possible to interpolate the return value of a registered "
                    "function, since variables are interpolated when the config "
                    "is loaded, and registered functions are resolved afterwards."
                )
                err = [{"loc": [parent, key], "msg": "uses section variable in string"}]
                raise ConfigValidationError("", err, message=err_title)

    def copy(self) -> "Config":
        """Deepcopy the config."""
        try:
            config = copy.deepcopy(self)
        except Exception as e:
            raise ValueError(f"Couldn't deep-copy config: {e}") from e
        return Config(
            config,
            is_interpolated=self.is_interpolated,
            section_order=self.section_order,
        )

    def merge(self, updates: Union[Dict[str, Any], "Config"]) -> "Config":
        """Deep merge the config with updates, using current as defaults."""
        defaults = self.copy()
        updates = Config(updates).copy()
        merged = deep_merge_configs(updates, defaults)
        return Config(
            merged,
            is_interpolated=defaults.is_interpolated and updates.is_interpolated,
            section_order=defaults.section_order,
        )

    def _sort(
        self, data: Union["Config", "ConfigParser", Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sort sections using the currently defined sort order. Sort
        sections by index on section order, if available, then alphabetic, and
        account for subsections, which should always follow their parent.
        """
        sort_map = {section: i for i, section in enumerate(self.section_order)}
        sort_key = lambda x: (sort_map.get(x[0].split(".")[0], len(sort_map)), x[0])
        return dict(sorted(data.items(), key=sort_key))

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
            config.set(section, option, dump_json(value, overrides))

    def _validate_sections(self, config: "ConfigParser") -> None:
        # If the config defines top-level properties that are not sections (e.g.
        # if config was constructed from dict), those values would be added as
        # [DEFAULTS] and included in *every other section*. This is usually not
        # what we want and it can lead to very confusing results.
        default_section = config.defaults()
        if default_section:
            err_title = "Found config values without a top-level section"
            err_msg = "not part of a section"
            err = [{"loc": [k], "msg": err_msg} for k in default_section]
            raise ConfigValidationError("", err, message=err_title)

    def from_str(
        self, text: str, *, interpolate: bool = True, overrides: Dict[str, Any] = {}
    ) -> "Config":
        """Load the config from a string."""
        config = get_configparser(interpolate=interpolate)
        config.read_string(text)
        config._sections = self._sort(config._sections)
        self._set_overrides(config, overrides)
        self.clear()
        self.interpret_config(config)
        self.is_interpolated = interpolate
        return self

    def to_str(self, *, interpolate: bool = True) -> str:
        """Write the config to a string."""
        flattened = get_configparser(interpolate=interpolate)
        queue: List[Tuple[tuple, "Config"]] = [(tuple(), self)]
        for path, node in queue:
            section_name = ".".join(path)
            is_kwarg = path and path[-1] != "*"
            if is_kwarg and not flattened.has_section(section_name):
                # Always create sections for non-'*' sections, not only if
                # they have leaf entries, as we don't want to expand
                # blocks that are undefined
                flattened.add_section(section_name)
            for key, value in node.items():
                if hasattr(value, "items"):
                    # Reference to a function with no arguments, serialize
                    # inline as a dict and don't create new section
                    if registry.is_promise(value) and len(value) == 1 and is_kwarg:
                        flattened.set(section_name, key, dump_json(value, node))
                    else:
                        queue.append((path + (key,), value))
                else:
                    flattened.set(section_name, key, dump_json(value, node))
        # Order so subsection follow parent (not all sections, then all subs etc.)
        flattened._sections = self._sort(flattened._sections)
        self._validate_sections(flattened)
        string_io = io.StringIO()
        flattened.write(string_io)
        return string_io.getvalue().strip()

    def to_bytes(self, *, interpolate: bool = True) -> bytes:
        """Serialize the config to a byte string."""
        return self.to_str(interpolate=interpolate).encode("utf8")

    def from_bytes(
        self,
        bytes_data: bytes,
        *,
        interpolate: bool = True,
        overrides: Dict[str, Any] = {},
    ) -> "Config":
        """Load the config from a byte string."""
        return self.from_str(
            bytes_data.decode("utf8"), interpolate=interpolate, overrides=overrides
        )

    def to_disk(self, path: Union[str, Path], *, interpolate: bool = True):
        """Serialize the config to a file."""
        path = Path(path)
        with path.open("w", encoding="utf8") as file_:
            file_.write(self.to_str(interpolate=interpolate))

    def from_disk(
        self,
        path: Union[str, Path],
        *,
        interpolate: bool = True,
        overrides: Dict[str, Any] = {},
    ) -> "Config":
        """Load config from a file."""
        with Path(path).open("r", encoding="utf8") as file_:
            text = file_.read()
        return self.from_str(text, interpolate=interpolate, overrides=overrides)


def dump_json(value: Any, data: Union[Dict[str, dict], Config, str] = "") -> str:
    """Dump a config value as JSON and output user-friendly error if it fails."""
    # Special case if we have a variable: it's already a string so don't dump
    # to preserve ${x:y} vs. "${x:y}"
    if isinstance(value, str) and VARIABLE_RE.search(value):
        return value
    try:
        return srsly.json_dumps(value)
    except Exception as e:
        err_msg = (
            f"Couldn't serialize config value of type {type(value)}: {e}. Make "
            f"sure all values in your config are JSON-serializable. If you want "
            f"to include Python objects, use a registered function that returns "
            f"the object instead."
        )
        raise ConfigValidationError(data, [], message=err_msg) from e


def deep_merge_configs(
    config: Union[Dict[str, Any], Config], defaults: Union[Dict[str, Any], Config],
) -> Union[Dict[str, Any], Config]:
    """Deep merge two configs."""
    for key, value in defaults.items():
        if isinstance(value, dict):
            node = config.setdefault(key, {})
            if not isinstance(node, dict):
                continue
            promises = [key for key in value if key.startswith("@")]
            promise = promises[0] if promises else None
            # We only update the block from defaults if it refers to the same
            # registered function
            if (
                promise
                and any(k.startswith("@") for k in node)
                and (promise in node and node[promise] != value[promise])
            ):
                continue
            defaults = deep_merge_configs(node, value)
        elif key not in config:
            config[key] = value
    return config


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


def alias_generator(name: str) -> str:
    """Generate field aliases in promise schema."""
    # Underscore fields are not allowed in model, so use alias
    if name == ARGS_FIELD_ALIAS:
        return ARGS_FIELD
    # Auto-alias fields that shadow base model attributes
    if name in RESERVED_FIELDS:
        return RESERVED_FIELDS[name]
    return name


class EmptySchema(BaseModel):
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


class _PromiseSchemaConfig:
    extra = "forbid"
    arbitrary_types_allowed = True
    alias_generator = alias_generator


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
    ) -> Tuple[Dict[str, Any], Config]:
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
        # If a Config was loaded with interpolate=False, we assume it needs to
        # be interpolated first, otherwise we take it at face value
        is_interpolated = not isinstance(config, Config) or config.is_interpolated
        section_order = config.section_order if isinstance(config, Config) else None
        orig_config = config
        if not is_interpolated:
            config = Config(orig_config).interpolate()
        filled, _, resolved = cls._fill(
            config, schema, validate=validate, overrides=overrides
        )
        filled = Config(filled, section_order=section_order)
        # Check that overrides didn't include invalid properties not in config
        if validate:
            cls._validate_overrides(filled, overrides)
        # Merge the original config back to preserve variables if we started
        # with a config that wasn't interpolated. Here, we prefer variables to
        # allow auto-filling a non-interpolated config without destroying
        # variable references.
        if not is_interpolated:
            filled = filled.merge(Config(orig_config, is_interpolated=False))
        return dict(resolved), filled

    @classmethod
    def make_from_config(
        cls,
        config: Union[Config, Dict[str, Dict[str, Any]]],
        *,
        schema: Type[BaseModel] = EmptySchema,
        overrides: Dict[str, Any] = {},
        validate: bool = True,
    ) -> Dict[str, Any]:
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
    ) -> Tuple[
        Union[Dict[str, Any], Config], Union[Dict[str, Any], Config], Dict[str, Any]
    ]:
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
            # If the field name is reserved, we use its alias for validation
            v_key = RESERVED_FIELDS.get(key, key)
            key_parent = f"{parent}.{key}".strip(".")
            if key_parent in overrides:
                value = overrides[key_parent]
                config[key] = value
            if cls.is_promise(value):
                promise_schema = cls.make_promise_schema(value)
                filled[key], validation[v_key], final[key] = cls._fill(
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
                    ) from err
                validation[v_key] = getter_result
                final[key] = getter_result
                if isinstance(validation[v_key], dict):
                    # The registered function returned a dict, prevent it from
                    # being validated as a config section
                    validation[v_key] = {}
                if isinstance(validation[v_key], GeneratorType):
                    # If value is a generator we can't validate type without
                    # consuming it (which doesn't work if it's infinite – see
                    # schedule for examples). So we skip it.
                    validation[v_key] = []
            elif hasattr(value, "items"):
                field_type = EmptySchema
                if key in schema.__fields__:
                    field = schema.__fields__[key]
                    field_type = field.type_
                    if not isinstance(field.type_, ModelMetaclass):
                        # If we don't have a pydantic schema and just a type
                        field_type = EmptySchema
                filled[key], validation[v_key], final[key] = cls._fill(
                    value,
                    field_type,
                    validate=validate,
                    parent=key_parent,
                    overrides=overrides,
                )
                if key == ARGS_FIELD and isinstance(validation[v_key], dict):
                    # If the value of variable positional args is a dict (e.g.
                    # created via config blocks), only use its values
                    validation[v_key] = list(validation[v_key].values())
                    final[key] = list(final[key].values())
            else:
                filled[key] = value
                # Prevent pydantic from consuming generator if part of a union
                validation[v_key] = (
                    value if not isinstance(value, GeneratorType) else []
                )
                final[key] = value
        # Now that we've filled in all of the promises, update with defaults
        # from schema, and validate if validation is enabled
        if validate:
            try:
                result = schema.parse_obj(validation)
            except ValidationError as e:
                raise ConfigValidationError(
                    config, e.errors(), element=parent
                ) from None
        else:
            # Same as parse_obj, but without validation
            result = schema.construct(**validation)
        exclude_validation = set([ARGS_FIELD_ALIAS, *RESERVED_FIELDS.keys()])
        validation.update(result.dict(exclude=exclude_validation))
        filled, final = cls._update_from_parsed(validation, filled, final)
        return filled, validation, final

    @classmethod
    def _update_from_parsed(
        cls, validation: Dict[str, Any], filled: Dict[str, Any], final: Dict[str, Any]
    ):
        """Update the final result with the parsed config like converted
        values recursively.
        """
        for key, value in validation.items():
            if key in RESERVED_FIELDS.values():
                continue  # skip aliases for reserved fields
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
                elif key in RESERVED_FIELDS.values():
                    continue
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
                name = RESERVED_FIELDS.get(param.name, param.name)
                sig_args[name] = (annotation, default)
        sig_args["__config__"] = _PromiseSchemaConfig
        return create_model("ArgModel", **sig_args)


__all__ = ["Config", "registry", "ConfigValidationError"]
