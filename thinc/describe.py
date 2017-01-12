class AttributeDescription(object):
    def __init__(self, name, value=None, *args, **kwargs):
        self.name = name
        self.value = value


class Dimension(AttributeDescription):
    def __call__(self, attr, model):
        '''Add the dimension to the instance.'''
        setattr(model, attr, None)


class Weights(AttributeDescription):
    def __init__(self, name, shape=None, init=None):
        self.name = name
        self.shape = shape
        self.init = init

    def __call__(self, attr, model):
        setattr(model, attr, None)


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
