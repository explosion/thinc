from functools import wraps


class AttributeDescription(object):
    def __init__(self, text, value=None, *args, **kwargs):
        self.name = None
        self.text = text
        self.value = value

    def __call__(self, attr, model):
        self.name = attr

    def __get__(self, obj, type=None): # pragma: no cover
        return self.value

    def __set__(self, obj, val): # pragma: no cover
        self.value = val


class Dimension(AttributeDescription):
    def __get__(self, obj, type=None):
        return obj._dims.get(self.name, None)

    def __set__(self, obj, value):
        obj._dims[self.name] = value


class Weights(AttributeDescription):
    def __init__(self, text, get_shape, init=None):
        self.name = None
        self.text = text
        self.get_shape = get_shape
        self.init = init

    def __get__(self, obj, type=None):
        key = (obj.id, self.name)
        if key in obj._mem:
            return obj._mem[key]
        else:
            shape = self.get_shape(obj)
            data = obj._mem.add(key, shape)
            if self.init is not None:
                self.init(data, obj.ops)
            return data

    def __set__(self, obj, val):
        data = obj._mem.get((obj.id, self.name))
        data[:] = val


class Gradient(AttributeDescription):
    def __init__(self, param_name):
        self.name = None
        self.text = "Gradient of %s" % param_name
        self.param_name = param_name

    def __get__(self, obj, type=None):
        key = (obj.id, self.name)
        if key in obj._mem:
            return obj._mem.get(key)
        else:
            param_key = (obj.id, self.param_name)
            grad = obj._mem.add_gradient(key, param_key)
            return grad

    def __set__(self, obj, val):
        data = obj._mem.get((obj.id, self.name))
        data[:] = val


class Synapses(Weights):
    pass


class Biases(Weights):
    pass


class Moment(Weights):
    pass


def attributes(**specs):
    if not specs: # pragma: no cover
        raise ValueError("Must describe at least one attribute")
    def wrapped(cls):
        cls.descriptions = dict(cls.descriptions)
        cls.descriptions.update(specs)
        for attr, desc in cls.descriptions.items():
            setattr(cls, attr, desc)
            desc.name = attr
        return cls
    return wrapped


def on_init(*callbacks):
    def wrapped(cls):
        cls.on_init_hooks = list(cls.on_init_hooks)
        cls.on_init_hooks.extend(callbacks)
        return cls
    return wrapped


def on_data(*callbacks):
    def wrapped(cls):
        cls.on_data_hooks = list(cls.on_data_hooks)
        cls.on_data_hooks.extend(callbacks)
        return cls
    return wrapped


def input(getter):
    def wrapped(cls):
        cls.describe_input = getter
        return cls
    return wrapped


def output(getter):
    def wrapped(cls):
        cls.describe_output = getter
        return cls
    return wrapped
