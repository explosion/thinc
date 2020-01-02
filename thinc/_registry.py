from typing import Callable, Dict, Any, Tuple, List, Optional
import catalogue
import inspect
import pydantic
from pydantic import BaseModel, create_model, ValidationError
from pydantic.main import ModelMetaclass
from wasabi import table


# TODO: pydantic's sequence_like includes generators, which causes Model.dict
# to fail if the values are generators. We need to find a solution for this.
pydantic.main.sequence_like = lambda v: isinstance(v, (list, tuple, set, frozenset))


class ConfigValidationError(ValueError):
    def __init__(self, config, errors):
        """Custom error for validating configs."""
        data = []
        for error in errors:
            err_loc = " -> ".join([str(p) for p in error.get("loc", [])])
            data.append((err_loc, error.get("msg")))
        result = ["Config validation error", table(data), f"{config}"]
        ValueError.__init__(self, "\n".join(result))


class _PromiseSchemaConfig:
    extra = "forbid"
    arbitrary_types_allowed = True


class EmptySchema(BaseModel):
    class Config:
        extra = "allow"


class registry(object):
    optimizers = catalogue.create("thinc", "optimizers", entry_points=True)
    schedules = catalogue.create("thinc", "schedules", entry_points=True)
    layers = catalogue.create("thinc", "layers", entry_points=True)

    @classmethod
    def create(cls, registry_name: str, entry_points: bool = False) -> None:
        """Create a new custom registry."""
        if hasattr(cls, registry_name):
            raise ValueError(f"Registry '{registry_name}' already exists")
        reg = catalogue.create("thinc", registry_name, entry_points=entry_points)
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
        config: Dict[str, Any],
        validate: bool = True,
        base_schema: Optional[ModelMetaclass] = EmptySchema,
    ) -> Dict[str, Any]:
        """Unpack a config dictionary, creating objects from the registry
        recursively.
        """
        # TODO: handle validation where top-level config is a promise
        if validate:
            filled, _ = cls.fill_and_validate(config, base_schema)
        else:
            filled = {}
        # Recurse over subdictionaries, filling in values.
        for key, value in config.items():
            if isinstance(value, dict):
                filled[key] = cls.make_from_config(value)
            else:
                filled[key] = value
        if cls.is_promise(filled):
            getter = cls.get_constructor(filled)
            args, kwargs = cls.parse_args(filled)
            return getter(*args, **kwargs)
        else:
            return filled

    @classmethod
    def fill_and_validate(cls, config, schema):
        """Build two representations of the config: one where the promises are
        preserved, and a second where the promises are represented by their
        return types. Use the validation representation to get default
        values via pydantic. The defaults are filled into both representations.
        """
        filled = {}
        validation = {}
        for key, value in config.items():
            if cls.is_promise(value):
                promise_schema = cls.make_promise_schema(value)
                filled[key], _ = cls.fill_and_validate(value, promise_schema)
                # Call the function and populate the field value. We can't just
                # create an instance of the type here, since this wouldn't work
                # for generics / more complex custom types
                getter = cls.get_constructor(filled[key])
                args, kwargs = cls.parse_args(filled[key])
                validation[key] = getter(*args, **kwargs)
            elif hasattr(value, "items"):
                if key in schema.__fields__:
                    field = schema.__fields__[key]
                    field_type = field.type_
                else:
                    field_type = EmptySchema
                filled[key], validation[key] = cls.fill_and_validate(value, field_type)
            else:
                filled[key] = value
                validation[key] = value
        # Now that we've filled in all of the promises, update with defaults
        # from schema, and validate.
        try:
            result = schema.parse_obj(validation)
        except ValidationError as e:
            raise ConfigValidationError(config, e.errors())
        validation.update(result.dict())
        for key, value in validation.items():
            if key not in filled:
                filled[key] = value
        return filled, validation

    @classmethod
    def is_promise(cls, obj: Any) -> bool:
        if not hasattr(obj, "keys"):
            return False
        id_keys = [k for k in obj.keys() if k.startswith("@")]
        if len(id_keys) != 1:
            return False
        else:
            return True

    @classmethod
    def get_constructor(cls, obj: Dict[str, Any]) -> Callable:
        id_keys = [k for k in obj.keys() if k.startswith("@")]
        if len(id_keys) != 1:
            err = f"A block can only contain one function registry reference. Got: {id_keys}"
            raise ValueError(err)
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
                if isinstance(key, int) or key.isdigit():
                    args.append((int(key), value))
                else:
                    kwargs[key] = value
        return [value for key, value in sorted(args)], kwargs

    @classmethod
    def make_promise_schema(cls, obj: Dict[str, Any]) -> ModelMetaclass:
        func = cls.get_constructor(obj)
        # Read the argument annotations and defaults from the function signature
        id_keys = [k for k in obj.keys() if k.startswith("@")]
        sig_args: Dict[str, Any] = {id_keys[0]: (str, ...)}
        for param in inspect.signature(func).parameters.values():
            # If no default value is specified assume that it's required
            if param.default != param.empty:
                sig_args[param.name] = (param.annotation, param.default)
            else:
                sig_args[param.name] = (param.annotation, ...)
        sig_args["__config__"] = _PromiseSchemaConfig
        return create_model("ArgModel", **sig_args)

    @classmethod
    def get_return_type(cls, obj: Dict[str, Any]):
        func = cls.get_constructor(obj)
        return inspect.signature(func).return_annotation
