class Dimensions(object):
    def __init__(self, **spec):
        for key, value in spec.items():
            if key.startswith('on_'):
                setattr(self, key, value)
            else:
                setattr(self, key, None)
                self.names[dim] = name


class Weights(object):
    def __init__(self, **spec):
        for key, values in spec.items():
            setattr(self, key, None)


def dimensions(**spec):
    if not spec:
        raise ValueError("Must describe at least one dimension")
    def wrapped(cls):
        cls.n = Dimensions(**spec)
        return cls
    return wrapped
            

def weights(**spec):
    if not spec:
        raise ValueError("Must describe at least one weight")
    def wrapped(cls):
        cls.w = Weights(**spec)
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
