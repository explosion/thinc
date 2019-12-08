# Configuration files

You can describe Thinc models and experiments using a configuration format
based on Python's built-in `configparser` module. We add a few additional
conventions to the built-in format, to provide support for non-string values,
support nested objects, and to integrate with Thinc's *registry system*.

## Example config


```
[some_section]
key1 = "string value"
another_key = 1.0
# Comments, naturally
third_key = ["values", "are parsed with", "json.loads()"]
some_other_key =  
    {
        "multiline values?": true
    }

[another_section]
more_values = "yes!"
null_values = null
interpolation = ${some_section:third_key}

# Describe nested sections with a dot notation in the section names.
[some_section.subsection]
# This will be moved, producing:
#   config["some_section"]["subsection"] = {"hi": true, "bye": false}
hi = true
bye = false
```

The config format has two has two main differences from the built-in `configparser`
module's behaviour:

* JSON-formatted values. Thinc passes all values through `json.loads()` to
  interpret them. You can use atomic values like strings, floats, integers,
  or booleans, or you can use complex objects such as lists or maps.

* Structured sections. Thinc uses a dot notation to build nested sections. If
  you have a section named `[outer_section.subsection]`, Thinc will parse that
  into a nested structure, placing `subsection` within `outer_section`

## Registry integration

Thinc's registry system lets you map string keys to functions. For instance,
let's say you want to define a new optimizer. You would define a function that
constructs it, and add it to the right register, like so:

```python

import thinc

@thinc.registry.optimizers.register("my_cool_optimizer.v1")
def make_my_optimizer(learn_rate, gamma):
    return MyCoolOptimizer(learn_rate, gamma)

# Later you can retrieve your function by name:
create_optimizer = thinc.registry.optimizers.get("my_cool_optimizer.v1")
```

The registry lets you refer to your function by string name, which is
often more convenient than passing around the function itself. This is
especially useful for configuration files: you can provide the name of your
function and the arguments in the config file, and you'll have everything you
need to rebuild the object.

Since this is a common workflow, the registry system provides a shortcut for
it, the `registry.make_from_config()` function. To use it, you just need to
follow a simple convention in your config file.

If a section contains a key beginning with @, the `registry.make_from_config()`
function will interpret the rest of that key as the name of the registry. The
value will be interpreted as the name to lookup. The rest of the section will
be passed to your function as arguments. Here's a simple example:

```
[optimizer]
@optimizers = "my_cool_optimizer.v1"
learn_rate = 0.001
gamma = 1e-8
```

The `registry.make_from_config()` function will fetch your 
`make_my_optimizer` function from the `optimizers` registry, call it using the
`learn_rate` and `gamma` arguments, and set the result of the function under
the key `"optimizer"`.

You can even use the  `registry.make_from_config()` function to build recursive
structures. Let's say your optimizer supports some sort of fancy visualisation
plug-in that Thinc has never heard of. All you would need to do is create a new
registry, named something like `visualizers`, and register a constructor
function, such as `my_visualizer.v1`. You would also make a new version of your
optimizer constructor, to pass in the new value. Now you can describe the
visualizer plugin in your config, so you can use it as an argument to your optimizer:

```
[optimizer]
@optimizers = "my_cool_optimizer.v2"
learn_rate = 0.001
gamma = 1e-8

[optimizer.visualizer]
@visualizers = "my_visualizer.v1"
format = "jpeg"
host = "localhost"
port = "8080"
```

The `optimizer.visualizer` section will be placed under the
`optimizer` object, using the key `visualizer` (see "structured sections"
above). The `registry.make_from_config()` function will build the visualizer
first, so that the result value is ready for the optimizer.
