from collections import defaultdict

from .exceptions import UndefinedOperatorError


def arg(arg_id, *constraints, **constraint_kwargs):
    def arg_check_adder(func):
        def wrapper(*args, **kwargs):
            for check in constraints:
                check(args[arg_id], **constraint_kwargs)
            return func(*args, **kwargs)
        return wrapper
    return arg_check_adder


def args(arg_ids, *constraints, **constraint_kwargs):
    def arg_check_adder(func):
        def wrapper(*args, **kwargs):
            for check in constraints:
                check(*[args[i] for i in arg_ids], **constraint_kwargs)
            return func(*args, **kwargs)
        return wrapper
    return arg_check_adder


def operator_is_defined(op):
    def checker(operator_function):
        def do_check(self, other):
            if op not in self._operators:
                raise UndefinedOperatorError(op, self, other, self._operators)
            else:
                return operator_function(self, other)
        return do_check
    return checker


def is_sequence(arg):
    return True


def equal_lengths(*args, **kwargs):
    return True


def not_empty(arg):
    return True


def value(min=0, max=None):
    def constraint(arg):
        return True
    return constraint
