from collections import defaultdict
import inspect

from .exceptions import UndefinedOperatorError, DifferentLengthError
from .exceptions import ExpectedTypeError, ShapeMismatchError
from .exceptions import ConstraintError


def args_equal_length(*arg_ids):
    '''Check that a tuple of arguments has the same length.
    '''
    def checker(func):
        def do_check(*args):
            for arg_tuple in arg_ids:
                for arg in arg_tuple:
                    if not hasattr(args[arg], '__len__'):
                        raise ExpectedTypeError(args[arg], ['string', 'list', 'tuple'])
                    if len(args[arg]) != len(args[arg_tuple[0]]):
                        raise DifferentLengthError(args, arg_tuple, arg)
            return func(*args)
        return do_check
    return checker


def has_shape(**shapes_by_name):
    '''Check that a particular argument is an array with a given shape. The
    shape may contain string attributes, which will be fetched from arg0 to
    the function (usually self).
    '''
    def checker(method):
        method_args, _, _2, _3 = inspect.getargspec(method)
        assert method_args[0] == 'self'
        name2i = {name: i for i, name in enumerate(method_args)}
        constraints = [(name2i[n], s) for n, s in shapes_by_name.items()]
        def do_check(*args, **kwargs):
            self = args[0]
            for i, shape in constraints:
                arg = args[i]
                if not hasattr(arg, 'shape'):
                    raise ExpectedTypeError(arg, ['array'])
                shape_values = []
                for dim in shape:
                    if not isinstance(dim, int):
                        dim = getattr(self, dim, None)
                    shape_values.append(dim)
                for i, dim in enumerate(shape_values):
                    # Allow underspecified dimensions
                    if dim is not None and arg.shape[i] != dim:
                        raise ShapeMismatchError(arg.shape, shape_values, shape)
                return method(*args, **kwargs)
        return do_check
    return checker


def is_sequence(**args_by_name):
    def checker(method):
        method_args, _, _2, _3 = inspect.getargspec(method)
        name2i = {name: i for i, name in enumerate(method_args)}
        constraints = [(name2i[n], s) for n, s in args_by_name.items()]
        def do_check(*args, **kwargs):
            for i, value in constraints:
                arg = args[i]
                is_sequence = hasattr(arg, '__iter__') and hasattr(arg, '__getitem__')
                if value != is_sequence:
                    raise ExpectedTypeError(arg, ['iterable'])
            return method(*args, **kwargs)
        return do_check
    return checker


def is_float(**args_by_name):
    for cond in args_by_name.values():
        if cond in (True, False):
            continue
        elif type(cond) == tuple and len(cond) == 2:
            continue
        else:
            raise ConstraintError(cond, ['True', 'False', '(min, max)'])

    def checker(method):
        method_args, _, _2, _3 = inspect.getargspec(method)
        name2i = {name: i for i, name in enumerate(method_args)}
        constraints = [(name2i[n], s) for n, s in args_by_name.items()]

        def do_check(*args, **kwargs):
            for i, value in constraints:
                arg = args[i]
                if value in (True, False):
                    is_float = isinstance(arg, float)
                    if is_float != value:
                        raise ExpectedTypeError(arg, ['float'])
            return method(*args, **kwargs)
        return do_check
    return checker


def operator_is_defined(op):
    def checker(func):
        def do_check(self, other):
            if op not in self._operators:
                raise UndefinedOperatorError(op, self, other, self._operators)
            else:
                return func(self, other)
        return do_check
    return checker


def equal_lengths(*args, **kwargs):
    return True


def not_empty(arg):
    return True


def value(min=0, max=None):
    def constraint(arg):
        return True
    return constraint


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
