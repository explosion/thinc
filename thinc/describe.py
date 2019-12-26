# TODO: These should probably be data classes?
class ParamInfo:
    """Information about a weights parameter. Stored in model._params"""

    def __init__(self, name, get_shape, init, text):
        self.name = name
        self.get_shape = get_shape
        self.init = init
        self.text = text


class GradInfo:
    """Information about a parameter gradient. Stored in model._grads"""

    def __init__(self, name, param_name, text):
        self.name = name
        self.param_name = param_name
        self.text = text


class AttributeDescription:
    def __init__(self, text, value=None, *args, **kwargs):
        self.name = None
        self.text = text
        self.value = value

    def install(self, attr, obj):
        self.name = attr

    def __call__(self, attr, obj):
        self.install(attr, obj)

    def __get__(self, obj, type=None):  # pragma: no cover
        return self.value

    def __set__(self, obj, val):  # pragma: no cover
        self.value = val


class Dimension(AttributeDescription):
    def install(self, attr, obj):
        self.name = attr
        obj._dims[self.name] = self.value

    def __get__(self, obj, type=None):
        return obj.get_dim(self.name)

    def __set__(self, obj, value):
        return obj.set_dim(self.name, value)


class Weights(AttributeDescription):
    def __init__(self, text, get_shape, init=None):
        self.name = None
        self.text = text
        self.get_shape = get_shape
        self.init = init

    def install(self, attr, obj):
        self.name = attr
        obj._params[self.name] = ParamInfo(
            self.name, self.get_shape, self.init, self.text
        )

    def __get__(self, obj, type=None):
        return obj.get_param(self.name)

    def __set__(self, obj, value):
        return obj.set_param(self.name, value)


class Gradient(AttributeDescription):
    def __init__(self, param_name):
        self.name = None
        self.text = "Gradient of %s" % param_name
        self.param_name = param_name

    def install(self, attr, obj):
        self.name = attr
        obj._grads[self.name] = GradInfo(self.name, self.param_name, self.text)

    def __get__(self, obj, type=None):
        return obj.get_grad(self.name)

    def __set__(self, obj, value):
        return obj.set_grad(self.name, value)


def attributes(**specs):
    if not specs:  # pragma: no cover
        raise ValueError("Must describe at least one attribute")

    def wrapped(cls):
        cls.descriptions = dict(cls.descriptions)
        cls.descriptions.update(specs)
        for attr, desc in cls.descriptions.items():
            setattr(cls, attr, desc)
            desc.name = attr
        return cls

    return wrapped
