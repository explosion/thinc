from collections import defaultdict, Sequence, Sized, Iterable
import inspect
import wrapt

from .exceptions import UndefinedOperatorError, DifferentLengthError
from .exceptions import ExpectedTypeError, ShapeMismatchError
from .exceptions import ConstraintError


def args_equal_length(*arg_ids):
    '''Check that a tuple of arguments has the same length.
    '''
    @wrapt.decorator
    def checker(wrapped, instance, args, kwargs):
        for arg_tuple in arg_ids:
            for arg in arg_tuple:
                if not isinstance(args[arg], Sized):
                    raise ExpectedTypeError(args[arg], ['Sized'])
                if len(args[arg]) != len(args[arg_tuple[0]]):
                    raise DifferentLengthError(args, arg_tuple, arg)
        return wrapped(*args, **kwargs)
    return checker


def has_shape(**args_by_name):
    '''Check that a particular argument is an array with a given shape. The
    shape may contain string attributes, which will be fetched from arg0 to
    the function (usually self).
    '''
    @wrapt.decorator
    def checker(wrapped, self, args, kwargs):
        constraints = _resolve_names(args_by_name,
            *inspect.getargspec(wrapped), **kwargs)
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
        return wrapped(*args, **kwargs)
    return checker


def is_sequence(**args_by_name):
    @wrapt.decorator
    def checker(wrapped, self, args, kwargs):
        constraints = _resolve_names(args_by_name, *inspect.getargspec(wrapped))
        for i, value in constraints:
            arg = args[i]
            if value != isinstance(arg, Iterable):
                raise ExpectedTypeError(arg, ['iterable'])
        return wrapped(*args, **kwargs)
    return checker


def is_float(**args_by_name):
    for cond in args_by_name.values():
        if isinstance(cond, bool) or (isinstance(cond, Sized) and len(cond) == 2):
            continue
        else:
            raise ConstraintError(cond, ['True', 'False', '(min, max)'])

    @wrapt.decorator
    def checker(wrapped, instance, args, kwargs):
        constraints = _resolve_names(args_by_name, *inspect.getargspec(wrapped))
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
    @wrapt.decorator
    def checker(wrapped, instance, args, kwargs):
        if instance is None:
            raise ExpectedTypeError(instance, ['Model'])
        if op not in instance._operators:
            raise UndefinedOperatorError(op, instance, args[0], instance._operators)
        else:
            return wrapped(*args, **kwargs)
    return checker


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


def _resolve_names(args_by_name, method_args, *_, **kwargs):
    has_self = 'self' in method_args
    name2i = {name: i-has_self for i, name in enumerate(method_args)}
    constraints = [(name2i[n], s) for n, s in args_by_name.items()]
    return constraints
