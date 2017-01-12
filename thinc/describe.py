class AttributeDescription(object):
    def __init__(self, text, value=None, *args, **kwargs):
        self.name = None
        self.text = text
        self.value = value

    def __call__(self, attr, model):
        self.name = attr

    def __get__(self, obj, type=None):
        return self.value

    def __set__(self, obj, val):
        self.value = val


class Dimension(AttributeDescription):
    def __get__(self, obj, type=None):
        return self.value

    def __set__(self, obj, value):
        self.value = value


class Weights(AttributeDescription):
    def __init__(self, text, shape=None, init=None):
        self.name = None
        self.text = text
        self.shape = shape
        self.init = init

    def __get__(self, obj, type=None):
        key = (obj.id, self.name)
        if key in obj.mem:
            return obj.mem[key]
        else:
            shape = tuple(getattr(obj, dim, None) for dim in self.shape)
            if any(dim is None for dim in shape):
                return None
            else:
                data = obj.mem.add(key, shape)
                if self.init is not None:
                    self.init(data, obj.ops)
                return data

    def __set__(self, obj, val):
        data = obj.mem.get((obj.id, self.name))
        data[:] = val


class Gradient(AttributeDescription):
    def __init__(self, param_name):
        self.name = None
        self.text = "Gradient of %s" % param_name
        self.param_name = param_name

    def __get__(self, obj, type=None):
        key = (obj.id, self.name)
        if key in obj.mem:
            return obj.mem.get(key)
        else:
            param_key = (obj.id, self.param_name)
            if param_key in obj.mem:
                grad = obj.mem.add_gradient(key, param_key)
                return grad
            else:
                return None
    
    def __set__(self, obj, val):
        data = obj.mem.get((obj.id, self.name))
        data[:] = val


class Synapses(Weights):
    pass


class Biases(Weights):
    pass


def attributes(**specs):
    if not specs:
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


def input(shape, **kwargs):
    def wrapped(cls):
        cls.input_shape = property(lambda self: map(self.n.get, shape))
        return cls
    return wrapped


def output(shape, **kwargs):
    def wrapped(cls):
        cls.output_shape = property(lambda self: map(self.n.get, shape))
        return cls
    return wrapped
