from collections import defaultdict

from .exceptions import UndefinedOperatorError


def arg(arg_id, *constraints):
    return ArgCheckAdder(arg_id, *constraints)


def args(arg_id, constraint, or_=None):
    return ArgCheckAdder(arg_id, constraint, or_)


def operator_is_defined(op):
    def checker(operator_function):
        def do_check(self, other):
            if op not in self._operators:
                raise UndefinedOperatorError(op, self, other, self._operators)
            else:
                return operator_function(self, other)
        return do_check
    return checker


def is_a(type_):
    pass


def length(min=0, max=None):
    pass


def value(min=0, max=None):
    pass


def equal(get_attr):
    pass


def match(get_attr):
    pass


class ArgCheckAdder(object):
    def __init__(self, arg_id, *constraints):
        self.arg_id = arg_id
        self.constraints = constraints

    def __call__(self, func):
        if hasattr(func, 'checks'):
            func.checks[arg_id].extend(self.constraints)
            return func
        else:
            return CheckedFunction(func, {self.arg_id: self.constraints})


class CheckedFunction(object):
    def __init__(self, func, checks):
        self.checks = defaultdict(list)
        self.checks.update(checks)
        self.func = func

    def __call__(self, *args, **kwargs):
        for arg_id, check in self.checks.items():
            check(self.get_args(arg_id, args, kwargs))
        self.func(*args, **kwargs)

