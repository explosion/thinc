---
title: Configuration System
next: /docs/usage-models
---

Configuration is a huge problem for machine-learning code because you may want
to expose almost any detail of any function as a hyperparameter. The setting you
want to expose might be arbitrarily far down in your call stack, so the setting
might need to pass all the way through the CLI or REST API, through any number
of intermediate functions, affecting the interface of everything along the way.
And then once those settings are added, they become hard to remove later.
Default values also become hard to change without breaking backwards
compatibility.

To solve this problem, Thinc provides a config system that lets you easily
describe **arbitrary trees of objects**. The objects can be created via
**function calls you register** using a simple decorator syntax. You can even
version the functions you create, allowing you to make improvements without
breaking backwards compatibility. The most similar config system we're aware of
is [Gin](https://github.com/google/gin-config), which uses a similar syntax, and
also allows you to link the configuration system to functions in your code using
a decorator. Thinc's config system is simpler and emphasizes a different
workflow via a subset of Gin's functionality.

<grid>

```ini
### Config {small="true"}
[training]
patience = 10
dropout = 0.2
use_vectors = false

[training.logging]
level = "INFO"

[nlp]
# This uses the value of training.use_vectors
use_vectors = ${training.use_vectors}
lang = "en"
```

```json
### Parsed {small="true"}
{
    "training": {
        "patience": 10,
        "dropout": 0.2,
        "use_vectors": false,
        "logging": {
            "level": "INFO"
        }
    },
    "nlp": {
        "use_vectors": false,
        "lang": "en"
    }
}
```

</grid>

The config is divided into sections, with the section name in square brackets –
for example, `[training]`. Within the sections, config values can be assigned to
keys using `=`. Values can also be referenced from other sections using the dot
notation and placeholders indicated by the dollar sign and curly braces. For
example, `${training.use_vectors}` will receive the value of `use_vectors` in
the `training` block. This is useful for settings that are shared across
components.

The config format has three main differences from Python's built-in
[`configparser`](https://docs.python.org/3/library/configparser.html):

1. **JSON-formatted values.** Thinc passes all values through `json.loads` to
   interpret them. You can use atomic values like strings, floats, integers or
   booleans, or you can use complex objects such as lists or maps.

2. **Structured sections.** Thinc uses a dot notation to build nested sections.
   If you have a section named `[section.subsection]`, Thinc will parse that
   into a nested structure, placing `subsection` within `section`.

3. **References to registry functions.** If a key starts with `@`, Thinc will
   interpret its value as the name of a function registry, load the function
   registered for that name and pass in the rest of the block as arguments. If
   type hints are available on the function, the argument values (and return
   value of the function) will be validated against them. This lets you express
   complex configurations, like a training pipeline where `batch_size` is
   populated by a function that yields floats (see
   [schedules](/docs/api-schedules)). Also see the section on
   [registry integration](#registry) for more details.

There's **no pre-defined scheme** you have to follow and how you set up the
top-level sections is up to you. At the end of it, you'll receive a dictionary
with the values that you can use in your script – whether it's complete
initialized functions, or just basic settings. For examples that show Thinc's
config system in action, check out the following tutorials:

<tutorials header="false">

- intro
- basic_cnn_tagger

</tutorials>

---

## Registry integration {#registry}

Thinc's [registry system](/docs/api-config#registry) lets you **map string keys
to functions**. For instance, let's say you want to define a new optimizer. You
would define a function that constructs it and add it to the right register,
like so:

```python
### Registering a function
from typing import Union, Iterable
import thinc

@thinc.registry.optimizers.register("my_cool_optimizer.v1")
def make_my_optimizer(learn_rate: Union[float, Iterable[float]], gamma: float):
    return MyCoolOptimizer(learn_rate, gamma)

# Later you can retrieve your function by name:
create_optimizer = thinc.registry.optimizers.get("my_cool_optimizer.v1")
```

The registry lets you refer to your function by string name, which is often more
convenient than passing around the function itself. This is especially useful
for configuration files: you can **provide the name of your function** and the
**arguments** in the config file, and you'll have everything you need to rebuild
the object.

Since this is a common workflow, the registry system provides a shortcut for it,
the [`registry.resolve`](/docs/api-config#registry-resolve) function. If a
section contains a key beginning with `@`, it will be interpreted as the name of
a function registry – e.g. `@optimizers` refers to a function registered in the
`optimizers` registry. The value will be interpreted as the name to look up and
the rest of the block will be passed into the function as arguments. Here's a
simple example:

<grid>

```ini
### config.cfg {small="true"}
[optimizer]
@optimizers = "my_cool_optimizer.v1"
learn_rate = 0.001
gamma = 1e-8
```

```python
### Usage {small="true"}
from thinc.api import Config, registry

config = Config().from_disk("./config.cfg")
resolved = registry.resolve(config)
optimizer = resolved["optimizer"]
```

</grid>

Under the hood, Thinc will look up the `"my_cool_optimizer.v1"` function in the
`"optimizers"` registry and then call it with the arguments `learn_rate` and
`gamma`. If the function has **type annotations**, it will also validate the
input. For instance, if `learn_rate` is annotated as a `float` and the config
defines a string, Thinc will raise an error.

```python
### Under the hood
optimizer_func = thinc.registry.get("optimizers", "my_cool_optimizer.v1")
optimizer = optimizer_func(learn_rate=0.001, gamma=1e-8)
```

<infobox variant="danger">

**Note on type annotations:** If type annotations only define basic types like
`str`, `int` or `bool`, the validation will accept all values that can be cast
to this type. For instance, `0` is considered valid for `bool`, since `bool(0)`
is valid, and the value passed to your function will be `False`. If you need
stricter validation, you can use
[`pydantic`'s strict types](https://pydantic-docs.helpmanual.io/usage/types/#strict-types)
For details and examples, see the
[section on advanced type annotations](#advanced-types) using
[`pydantic`](https://github.com/samuelcolvin/pydantic).

</infobox>

### Recursive blocks {#registry-recursive}

The function registry integration becomes even more powerful when used to build
**recursive structures**. Let's say you want to use a learning rate schedule and
pass in a generator as the `learn_rate` argument. Here's an example of a
function that yields an infinite series of decaying values, following the
schedule `base_rate * 1 / (1 + decay * t)`. It's also available in Thinc as
[`schedules.decaying`](/docs/api-schedules#decaying). The decorator registers
the function `"my_cool_decaying_schedule.v1"` in the registry `schedules`:

```python
from typing import Iterable
import thinc

@thinc.registry.schedules("my_cool_decaying_schedule.v1")
def decaying(base_rate: float, decay: float, *, t: int = 0) -> Iterable[float]:
    while True:
        yield base_rate * (1.0 / (1.0 + decay * t))
        t += 1
```

In your config, you can now define the `learn_rate` as a subsection of
`optimizer`, and specify its registry function and arguments:

```ini
### config.cfg
[optimizer]
@optimizers = "my_cool_optimizer.v1"
gamma = 1e-8

[optimizer.learn_rate]
@schedules = "my_cool_decaying_schedule.v1"
base_rate = 0.001
decay = 1e-4
```

When Thinc resolves the config, it will first look up
`"my_cool_decaying_schedule.v1"` and call it with its arguments. Both arguments
will be validated against the type annotations (<tt>float</tt>). The return
value will then be passed to the optimizer function as the `learn_rate`
argument. If type annotations are available for the return value and it's a type
that can be evaluated, the return value of the function will be validated as
well.

<infobox variant="warning">

**A note on validating generators:** If a value is a generator, it won't be
validated further, since this would mean having to execute and consume it.
Generators can potentially be infinite – like the decaying schedule in this
example – so checking its return value isn't viable.

</infobox>

```python
### Under the hood
learn_rate_func = thinc.registry.get("schedules", "my_cool_decaying_schedule.v1")
learn_rate = learn_rate_func(base_rate=0.001, decay=1e-4)

optimizer_func = thinc.registry.get("optimizers", "my_cool_optimizer.v1")
optimizer = optimizer_func(learn_rate=learn_rate, gamma=1e-8)
```

After resolving the config and filling in the values, `registry.resolve` will
return a tuple of the resolved config and the filled config with default values
added. The resolved config will be a dict with one key, `"optimizer"`, mapped to
an instance of the custom optimizer function initialized with the arguments
defined in the config.

<grid>

```python
### Usage {small="true"}
from thinc.api import Config, registry

config = Config().from_disk("./config.cfg")
resolved = registry.resolve(config)
```

```python
### Result {small="true"}
{
    "optimizer": <MyCoolOptimizer>
}
```

</grid>

### Defining variable positional arguments {#registries-args}

If you're setting function arguments in a config block, Thinc will expect the
function to have an argument of that same name. For instance,
`base_rate = 0.001` means that the function will be called with
`base_rate=0.001`. This works fine, since Python allows function arguments to be
supplied as positional arguments _or_ as keyword arguments. If possible, named
arguments are recommended, since it makes your code and config more explicit.

However, in some situations, your registered function may accept variable
positional arguments. In your config, you can then use `*` to define a list of
values:

<grid>

```python
### {small="true"}
@thinc.registry.schedules("my_cool_schedule.v1")
def schedule(*steps: float, final: float = 1.0) -> Iterable[float]:
    yield from steps
    while True:
        yield final
```

```ini
### config.cfg {small="true"}
[schedule]
@schedules = "my_cool_schedule.v1"
* = [0.05, 0.1, 0.25, 0.75, 0.9]
final = 1.0
```

</grid>

<infobox variant="warning">

**About type hints for variable arguments**: Type hints for variable arguments
should always describe the
[type of the individual arguments](https://www.python.org/dev/peps/pep-0484/#arbitrary-argument-lists-and-default-argument-values).
If your arguments are floats, you'd annotate them as `*args: float` (even though
`args` is technically a tuple of floats). This is also how Thinc will validate
the config: if variable arguments are annotated as `float`, each item in `*`
will be validated against `float`.

</infobox>

You can also use the `*` placeholder in nested configs to populate positional
arguments from function registries. This is useful for combinators like
[`chain`](/docs/api-layers#combinators) that take a variable number of layers as
arguments. The following config will create two [`Relu`](/docs/api-layers#relu)
layers, pass them to [`chain`](/docs/api-layers#combinators) and return a
combined model:

<grid>

```ini
### Config {small="true"}
[model]
@layers = "chain.v1"

[model.*.relu1]
@layers = "Relu.v1"
nO = 512
dropout = 0.2

[model.*.relu2]
@layers = "Relu.v0"
nO = 256
dropout = 0.1
```

```python
### Equivalent to {small="true"}
from thinc.api import chain, Relu

model = chain(
    Relu(nO=512, dropout=0.2),
    Relu(nO=256, dropout=0.1)
)
```

</grid>

### Using interpolation {#config-interpolation}

For hyperparameters and other settings that need to be used in different places
across your config, you can define a separate block once and then reference the
values using the
[extended interpolation](https://docs.python.org/3/library/configparser.html#configparser.ExtendedInterpolation).
For example, `${hyper_params.dropout}` will insert the value of `dropout` from
the section `hyper_params`.

<grid>

```ini
### config.cfg {small="true"}
[hyper_params]
hidden_width = 512
dropout = 0.2

[model]
@layers = "Relu.v1"
nO = ${hyper_params.hidden_width}
dropout = ${hyper_params.dropout}
```

```json
### Parsed {small="true"}
{
    "hyper_params": {
        "hidden_width": 512,
        "dropout": 0.2
    },
    "model": {
        "@layers": "Relu.v1",
        "nO": 512,
        "dropout": 0.2
    }
}
```

</grid>

<infobox variant="warning">

Interpolation happens **when the config is parsed**, i.e. when you create a
[`Config`](/docs/api-config#config) object from a file, string, etc. and
**before registered functions are resolved**. This means that you can only use
it for hard-coded values like numbers and strings, not to define placeholders
for resolved values. For instance, the `[model]` section in this example will be
resolved to an instance of `Relu` – but you wouldn't be able to use `${model}`
as the argument to another function defined in the config. To achieve this, you
can either define another nested block or
[variable positional arguments](#registries-args).

</infobox>

### Using custom registries {#registries-custom}

Thinc's `registry` includes
[several pre-defined registries](/docs/api-config#registry) that are also used
for its built-in functions. You can also use the
[`registry.create`](/docs/api-config#registry-create) method to add your own
registries that you can then reference in config files. The following will
create a registry `visualizers` and let you use the
`@thinc.registry.visualizers` decorator, as well as the `@visualizers` key in
config files.

```python
### Example
import thinc

thinc.registry.create("visualizers")

@thinc.registry.visualizers("my_cool_visualizer.v1")
def my_cool_visualizer(file_format: str = "jpg"):
    return MyCoolVisualizer(file_format)
```

<grid>

```ini
### config.cfg {small="true"}
[visualizer]
@visualizers = "my_cool_visualizer.v1"
file_format = "svg"
```

```python
### Result {small="true"}
{
    "visualizer": <MyCoolVisualizer>
}
```

</grid>

---

## Advanced type annotations with Pydantic {#advanced-types}

<infobox>

If you're new to type hints and Python types, check out this great introduction
from the [FastAPI docs](https://fastapi.tiangolo.com/python-types/).

</infobox>

[`pydantic`](https://github.com/samuelcolvin/pydantic) is a modern Python
library for **data parsing and validation** using type hints. It's used by Thinc
to validate configuration files, and you can also use it in your model and
component definition to enforce stricter and more fine-grained validation. If
type annotations only define basic types like `str`, `int` or `bool`, the
validation will accept all values that can be cast to this type. For instance,
`0` is considered valid for `bool`, since `bool(0)` is valid. If you need
stricter validation, you can use
[strict types](https://pydantic-docs.helpmanual.io/usage/types/#strict-types)
instead. This example defines an optimizer that only accepts a float, a positive
integer and a constrained string matching the given regular expression:

```python
### Example {highlight="2,6-8"}
import thinc
from pydantic import StrictFloat, PositiveInt, constr

@thinc.registry.optimizers("my_cool_optimizer.v1")
def my_cool_optimizer(
    learn_rate: StrictFloat,
    steps: PositiveInt = 10,
    log_level: constr(regex="(DEBUG|INFO|WARNING|ERROR)") = "ERROR"
):
    return MyCoolOptimizer(learn_rate, steps, log_level)
```

If your config defines a value that's not compatible with the type annotations –
for instance, a negative integer for `steps` – Thinc will raise an error:

<grid>

```ini
### config.cfg {small="true"}
[optimizer]
@optimizers = "my_cool_optimizer.v1"
learn_rate = 0.001
steps = -1
log_level = "DEBUG"
```

```
### Errors {small="true" wrap="true"}
Config validation error

steps   ensure this value is greater than 0

{'@optimizers': 'my_cool_optimizer.v1', 'learn_rate': 0.001, 'steps': -1, 'log_level': 'DEBUG'}
```

</grid>

<infobox>

For an overview of the available custom types, see the
[`pydantic` documentation](https://pydantic-docs.helpmanual.io/usage/types/).

</infobox>

Argument annotations can also define
[`pydantic` models](https://pydantic-docs.helpmanual.io/usage/models/). This is
useful if your function takes dictionaries as arguments. The data is then passed
to the model and is parsed and validated. `pydantic` models are classes that
inherit from the `pydantic.BaseModel` class and define fields with type hints
and optional defaults as attributes:

```python
### Example {highlight="3-6,12"}
from pydantic import BaseModel, StrictStr, constr, StrictBool

class LoggingConfig(BaseModel):
    name: StrictStr
    level: constr(regex="(DEBUG|INFO|WARNING|ERROR)") = "INFO"
    use_colors: StrictBool = True

@thinc.registry.optimizers("my_cool_optimizer.v1")
def my_cool_optimizer(
    learn_rate: StrictFloat,
    steps: PositiveInt = 10,
    logging_config: LoggingConfig
):
    return MyCoolOptimizer(learn_rate, steps, logging_config)
```

In the config file, `logging_config` can now become its own section,
`[optimizer.logging_config]`. Its values will be validated against the
`LoggingConfig` schema:

```ini
### config.cfg
[optimizer]
@optimizers = "my_cool_optimizer.v1"
learn_rate = 0.001
steps = 100

[optimizer.logging_config]
name = "my_logger"
level = "DEBUG"
use_colors = false
```

For even more flexible validation of values and relationships between them, you
can define [validators](https://pydantic-docs.helpmanual.io/usage/validators/)
that apply to one or more attributes and return the parsed attribute. In this
example, the validator checks that the value of `name` doesn't contain spaces
and returns its lowercase form:

```python
### Example {highlight="8-12"}
from pydantic import BaseModel, StrictStr, constr, StrictBool, validator

class LoggingConfig(BaseModel):
    name: StrictStr
    level: constr(regex="(DEBUG|INFO|WARNING|ERROR)") = "INFO"
    use_colors: StrictBool = True

    @validator("name")
    def validate_name(cls, v):
        if " " in v:
            raise ValueError("name can't contain spaces")
        return v.lower()
```

### Using a base schema {#advanced-types-base-schema}

If a config file specifies registered functions, their argument values will be
validated against the type annotations of the function. For all other values,
you can pass a `schema` to
[`registry.resolve`](/docs/api-config#registry-resolve), a
[`pydantic` model](https://pydantic-docs.helpmanual.io/usage/models/) used to
parse and validate the data. Models can also be nested to describe nested
objects.

```python
### Schema
from pydantic import BaseModel, StrictInt, StrictFloat, StrictBool, StrictStr
from typing import List

class TrainingSchema(BaseModel):
    patience: StrictInt
    dropout: StrictFloat
    use_vectors: StrictBool = False

class NlpSchema(BaseModel):
    lang: StrictStr
    pipeline: List[StrictStr]

class ConfigBaseSchema(BaseModel):
    training: TrainingSchema
    nlp: NlpSchema

    class Config:
        extra = "forbid"
```

Setting `extra = "forbid"` in the
[`Config`](https://pydantic-docs.helpmanual.io/usage/model_config/) means that
validation will fail if the object contains additional properties – for
instance, another top-level section that's not `training`. The default value,
`"ignore"`, means that additional properties will be ignored and filtered out.
Setting `extra = "allow"` means any extra values will be passed through without
validation.

<grid>

```ini
### config.cfg {small="true"}
[training]
patience = 10
dropout = 0.2
use_vectors = false

[nlp]
lang = "en"
pipeline = ["tagger", "parser"]
```

```python
### Usage {small="true" highlight="6"}
from thinc.api import registry, Config

config = Config().from_disk("./config.cfg")
resolved = registry.resolve(
    config,
    schema=ConfigBaseSchema
)
```

</grid>

### Filling a config with defaults {#advanced-types-fill-defaults}

The main motivation for Thinc's configuration system was to eliminate hidden
defaults and ensure that config settings are passed around consistently. This
also means that config files should always define **all available settings**.
The [`registry.fill`](/docs/api-config#registry-fill) method also
resolves the config, but it leaves references to registered functions intact and
doesn't replace them with their return values. If type annotations and/or a base
schema are available, they will be used to parse the config and fill in any
missing values and defaults to create an up-to-date "master config".

Let's say you've updated your schema and scripts to use two additional optional
settings. These settings should also be reflected in your config files so they
accurately represent the available settings (and don't assume any hidden
defaults).

```python
### {highlight="7-8"}
from pydantic import BaseModel, StrictInt, StrictFloat, StrictBool

class TrainingSchema(BaseModel):
    patience: StrictInt
    dropout: StrictFloat
    use_vectors: StrictBool = False
    use_tok2vec: StrictBool = False
    max_epochs: StrictInt = 100
```

Calling [`registry.fill`](/docs/api-config#registry-fill) with your
existing config will produce an updated version of it including the new settings
and their defaults:

<grid>

```ini
### Before {small="true"}
[training]
patience = 10
dropout = 0.2
use_vectors = false
```

```ini
### After {small="true" highlight="5-6"}
[training]
patience = 10
dropout = 0.2
use_vectors = false
use_tok2vec = false
max_epochs = 100
```

</grid>

The same also works for config blocks that reference registry functions. If your
**function arguments change**, you can run `registry.fill` to get your config up
to date with the new defaults. For instance, let's say the optimizer now allows
a new setting, `gamma`, that defaults to `1e-8`:

```python
### Example {highlight="8"}
import thinc
from pydantic import StrictFloat, PositiveInt, constr

@thinc.registry.optimizers("my_cool_optimizer.v2")
def my_cool_optimizer_v2(
    learn_rate: StrictFloat,
    steps: PositiveInt = 10,
    gamma: StrictFloat = 1e-8,
    log_level: constr(regex="(DEBUG|INFO|WARNING|ERROR)") = "ERROR"
):
    return MyCoolOptimizer(learn_rate, steps, gamma, log_level)
```

The config file should now also reflect this new setting and the default value
that's being passed in – otherwise, you'll lose that piece of information.
Running `registry.fill` solves this and returns a new `Config` with the complete
set of available settings:

<grid>

```ini
### Before {small="true"}
[optimizer]
@optimizers = "my_cool_optimizer.v2"
learn_rate = 0.001
steps = 100
log_level = "INFO"
```

```ini
### After {small="true" highlight="5"}
[optimizer]
@optimizers = "my_cool_optimizer.v2"
learn_rate = 0.001
steps = 100
gamma = 1e-8
log_level = "INFO"
```

</grid>

<infobox variant="warning">

Note that if you're only filling and not resolving a config, Thinc will **not**
load or call any registered functions, and it won't be able to validate the
return values of registered functions against any types defined in the base
schema. If you neeed to check that all functions exist and their return values
match, use [`registry.resolve`](/docs/api-config#registry-resolve) instead.

</infobox>

<!-- TODO:

---

## Examples {#examples}

<tabs id="example-config">
<tab title="Tagging & parsing with multi-task BiLSTM">

```ini
https://github.com/explosion/spaCy/blob/feature/config/examples/experiments/ptb-joint-pos-dep/bilstm_tok2vec.cfg
```

</tab>
</tabs>

more examples? or do examples differently? -->
