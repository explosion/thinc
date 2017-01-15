from __future__ import print_function
from collections import defaultdict
from thinc.exceptions import ExpectedIntError


def check_arg(arg_id, *constraints):
    return ArgCheckAdder(arg_id, *constraints)


def check_int_value(min=None, max=None):
    def constraint(arg_id, args, kwargs):
        if not isinstance(args[arg_id], int):
            raise ExpectedIntError(args[arg_id])
        assert args[arg_id] >= 0
    return constraint


class ArgCheckAdder(object):
    def __init__(self, arg_id, *constraints):
        self.arg_id = arg_id
        self.constraints = constraints

    def __call__(self, func):
        if hasattr(func, 'checks'):
            func.checks[self.arg_id].extend(self.constraints)
            return func
        else:
            return CheckedFunction(func, {self.arg_id: self.constraints})


class CheckedFunction(object):
    def __init__(self, func, checks):
        self.checks = defaultdict(list)
        self.checks.update(checks)
        self.func = func

    def __call__(self, *args, **kwargs):
        for arg_id, checks in self.checks.items():
            for check in checks:
                check(arg_id, args, kwargs)
        self.func(*args, **kwargs)


@check_arg(0, check_int_value(min=0))
@check_arg(1, check_int_value(min=0))
def add_positive_integers(int1, int2):
    return int1 + int2


def main():
    print(add_positive_integers(None, 5))
    print(add_positive_integers(0, 5))
    print(add_positive_integers(-1, 5))


if __name__ == '__main__':
    main()
