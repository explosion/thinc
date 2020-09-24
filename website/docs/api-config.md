---
title: Config & Registry
teaser: Function registry and configuration files
next: /docs/api-types
---

|                           |                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------- |
| [**Registry**](#registry) | Function registry for layers, optimizers etc.                                   |
| [**Config**](#config)     | `Config` class used to load and create INI-style [configs](/docs/usage-config). |

## Registry {#registry tag="class"}

Thinc's registry system lets you **map string keys to functions**. You can
register functions to create [optimizers](/docs/api-optimizers),
[schedules](/docs/api-schedules), [layers](/docs/api-layers) and more, and then
refer to them and set their arguments in your [config file](/docs/usage-config).
Python type hints are used to validate the inputs.

```python
### Example
import thinc

@thinc.registry.optimizers.register("my_cool_optimizer.v1")
def make_my_optimizer(learn_rate: float, gamma: float):
    return MyCoolOptimizer(learn_rate, gamma)
```

<grid>

```ini
### Valid Config {small="true"}
[optimizer]
@optimizers = "my_cool_optimizer.v1"
learn_rate = 0.001
gamma = 1e-8
```

```ini
### Invalid Config {small="true"}
[optimizer]
@optimizers = "my_cool_optimizer.v1"
learn_rate = 1  # not a float
schedules = null  # unknown argument
```

</grid>

### Attributes {#registry-attributes}

| Registry name | Description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `optimizers`  | Registry for functions that create [optimizers](/docs/api-optimizers). |
| `schedules`   | Registry for functions that create [schedules](/docs/api-schedules).   |
| `layers`      | Registry for functions that create [layers](/docs/api-layers).         |

### registry.get {#registry-get tag="classmethod"}

Get a registered function from a given registry using string names. Will raise
an error if the registry or function doesn't exist. All individual registries
also have a `get` method to get a registered function.

```python
### Example
registered_func = thinc.registry.get("optimizers", "my_cool_optimizer.v1")
# The above is the same as:
registered_func = thinc.registry.optimizers.get("my_cool_optimizer.v1")
```

| Argument        | Type              | Description                                    |
| --------------- | ----------------- | ---------------------------------------------- |
| `registry_name` | <tt>str</tt>      | The name of the registry, e.g. `"optimizers"`. |
| `func_name`     | <tt>str</tt>      | The name of the function.                      |
| **RETURNS**     | <tt>Callable</tt> | The registered function.                       |

### registry.create {#registry-create tag="classmethod"}

Create a new function registry that will become available as an attribute to
`registry`. Will raise an error if a registry of the name already exists. Under
the hood, this calls into
[`catalogue.create`](https://github.com/explosion/catalogue#function-cataloguecreate)
using the `"thinc"` namespace.

```python
### Example
thinc.registry.create("visualizers")

@thinc.registry.visualizers("my_cool_visualizer.v1")
def my_cool_visualizer(format: str = "jpg") -> "MyCoolVisualizer":
    return MyCoolVisualizer(format)
```

| Argument        | Type          | Description                                                                                                                                                    |
| --------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `registry_name` | <tt>str</tt>  | The name of the registry to create, e.g. `"visualizers"`.                                                                                                      |
| `entry_points`  | <tt>bool</tt> | Allow the registry to be populated with entry points advertised by other packages (e.g. via the `"thinc_visualizers"` entry point group). Defaults to `False`. |

<infobox variant="warning">

Registry names can be _any string_ – however, if you want to use your registry
as an attribute of `thinc.registry`, e.g. `@thinc.registry.visualizers`, they
should be valid Python attribute names and only contain alphanumeric characters
and underscores.

</infobox>

### registry.make_from_config {#registry-make_from_config tag="classmethod"}

Unpack a config dictionary, creating objects from the registry recursively. If a
section contains a key beginning with `@`, the the rest of that key will be
interpreted as the name of the registry. For instance,
`"@optimizers": "my_cool_optimizer.v1"` will load the function from the
optimizers registry and pass in the specified arguments. For more details and
examples, see the [docs on configuration files](/docs/usage-config).

```python
### Example
from thinc.api import Config

cfg = Config().from_disk("./my_config.cfg")
loaded_cfg = registry.make_from_config(cfg)
```

| Argument       | Type                                   | Description                                                                                                                                                                                                                                                                             |
| -------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`       | <tt>Union[Config, Dict[str, Any]]</tt> | The config dict to load.                                                                                                                                                                                                                                                                |
| _keyword-only_ |                                        |                                                                                                                                                                                                                                                                                         |
| `validate`     | <tt>bool</tt>                          | Whether to validate the config against a base schema and/or type annotations defined on the registered functions. Defaults to `True`.                                                                                                                                                   |
| `schema`       | <tt>pydantic.BaseModel</tt>            | Optional [`pydantic` model](https://pydantic-docs.helpmanual.io/usage/models/) to validate the config against. See the docs on [base schemas](/docs/api-config#advanced-types-base-schema) for details. Defaults to an `EmptySchema` with extra properties and arbitrary types allowed. |
| **RETURNS**    | <tt>Config</tt>                        | The filled config.                                                                                                                                                                                                                                                                      |

### registry.fill_config {#fill_config tag="classmethod"}

Unpack a config dictionary, but leave all references to registry functions
intact and don't resolve them. Only use the type annotations and optional base
schema to fill in all arguments and their default values. This method is
especially useful for getting an existing config up to date with changes in the
schema and/or function arguments. If the config is incomplete and contains
missing values for required arguments, you can set `validate=False` to skip
validation and only update it. The updated schema should then pass validation.

```python
### Example
from thinc.api import Config

cfg = Config().from_disk("./my_config.cfg")
filled_cfg = registry.fill_config(cfg)
```

| Argument       | Type                                   | Description                                                                                                                                                                                                                                                                             |
| -------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`       | <tt>Union[Config, Dict[str, Any]]</tt> | The config dict to load.                                                                                                                                                                                                                                                                |
| _keyword-only_ |                                        |                                                                                                                                                                                                                                                                                         |
| `validate`     | <tt>bool</tt>                          | Whether to validate the config against a base schema and/or type annotations defined on the registered functions. Defaults to `True`.                                                                                                                                                   |
| `schema`       | <tt>pydantic.BaseModel</tt>            | Optional [`pydantic` model](https://pydantic-docs.helpmanual.io/usage/models/) to validate the config against. See the docs on [base schemas](/docs/api-config#advanced-types-base-schema) for details. Defaults to an `EmptySchema` with extra properties and arbitrary types allowed. |
| **RETURNS**    | <tt>Config</tt>                        | The filled config.                                                                                                                                                                                                                                                                      |

---

## Config {#config tag="class"}

This class holds the model and training [configuration](/docs/usage-config) and
can load and save the INI-style configuration format from/to a string, file or
bytes. The `Config` class is a subclass of `dict` and uses Python's
`ConfigParser` under the hood.

### Config.\_\_init\_\_ {#config-init tag="method"}

Initialize a new `Config` object with optional data.

| Argument | Type                                             | Description                                  |
| -------- | ------------------------------------------------ | -------------------------------------------- |
| `data`   | <tt>Optional[Union[Dict[str, Any], Config]]</tt> | Optional data to initialize the config with. |

### Config.from_str {#config-from_str tag="method"}

Load the config from a string.

```python
### Example
from thinc.api import Config

config_str = """
[training]
patience = 10
dropout = 0.2
"""
config = Config().from_str(config_str)
print(config["training"])  # {'patience': 10, 'dropout': 0.2}}
```

| Argument    | Type            | Description                |
| ----------- | --------------- | -------------------------- |
| `text`      | <tt>str</tt>    | The string config to load. |
| **RETURNS** | <tt>Config</tt> | The loaded config.         |

### Config.to_str {#config-to_str tag="method"}

Write the config to a string.

```python
### Example
from thinc.api import Config

config = Config({"training": {"patience": 10, "dropout": 0.2}})
print(config.to_str()) # '[training]\npatience = 10\n\ndropout = 0.2'
```

| Argument    | Type         | Description        |
| ----------- | ------------ | ------------------ |
| **RETURNS** | <tt>str</tt> | The string config. |

### Config.to_bytes {#config-to_bytes tag="method"}

Serialize the config to a byte string.

```python
### Example
from thinc.api import Config

config = Config({"training": {"patience": 10, "dropout": 0.2}})
config_bytes = config.to_bytes()
print(config_bytes)  # b'[training]\npatience = 10\n\ndropout = 0.2'
```

| Argument    | Type           | Description            |
| ----------- | -------------- | ---------------------- |
| **RETURNS** | <tt>bytes</tt> | The serialized config. |

### Config.from_bytes {#config-from_bytes tag="method"}

Load the config from a byte string.

```python
### Example
from thinc.api import Config

config = Config({"training": {"patience": 10, "dropout": 0.2}})
config_bytes = config.to_bytes()
new_config = Config().from_bytes(config_bytes)
```

| Argument     | Type            | Description        |
| ------------ | --------------- | ------------------ |
| `bytes_data` | <tt>bytes</tt>  | The data to load.  |
| **RETURNS**  | <tt>Config</tt> | The loaded config. |

### Config.to_disk {#config-to_disk tag="method"}

Serialize the config to a file.

```python
### Example
from thinc.api import Config

config = Config({"training": {"patience": 10, "dropout": 0.2}})
config.to_disk("./config.cfg")
```

| Argument | Type                      | Description    |
| -------- | ------------------------- | -------------- |
| `path`   | <tt>Union[Path, str]</tt> | The file path. |

### Config.from_disk {#config-from_disk tag="method"}

Load the config from a file.

```python
### Example
from thinc.api import Config

config = Config({"training": {"patience": 10, "dropout": 0.2}})
config.to_disk("./config.cfg")
new_config = Config().from_disk("./config.cfg")
```

| Argument    | Type                      | Description        |
| ----------- | ------------------------- | ------------------ |
| `path`      | <tt>Union[Path, str]</tt> | The file path.     |
| **RETURNS** | <tt>Config</tt>           | The loaded config. |
