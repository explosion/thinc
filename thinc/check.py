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

def noop(func, *args, **kwargs):
    return func

def is_a(type_):
    def constraint(arg_id, args, kwargs):
        return True
    return constraint



def length(min=0, max=None):
    def constraint(arg_id, args, kwargs):
        return True
    return constraint


def value(min=0, max=None):
    def constraint(arg_id, args, kwargs):
        return True
    return constraint


def equal(get_attr):
    def constraint(arg_id, args, kwargs):
        return True
    return constraint


def match(get_attr):
    def constraint(arg_id, args, kwargs):
        return True
    return constraint


class ArgCheckAdder(object):
    def __init__(self, arg_id, *constraints):
        self.arg_id = arg_id
        self.constraints = list(constraints)

    def __call__(self, func):
        if hasattr(func, 'checks'):
            func.checks[self.arg_id].extend(self.constraints)
            return func
        else:
            return CheckedFunction(func, {self.arg_id: self.constraints})


class CheckedFunction(object):
    def __init__(self, func, checks):
        for check in checks.values():
            for c in check:
                assert check is not None
        self.checks = defaultdict(list)
        self.checks.update(checks)
        self.func = func

    def __call__(self, *args, **kwargs):
        for arg_id, checks in self.checks.items():
            for check in checks:
                check(arg_id, args, kwargs)
        return self.func(*args, **kwargs)


