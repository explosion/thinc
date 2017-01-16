from collections import defaultdict, Sequence, Sized, Iterable
import inspect
import wrapt
from cytoolz import curry
from numpy import ndarray

from .exceptions import UndefinedOperatorError, DifferentLengthError
from .exceptions import ExpectedTypeError, ShapeMismatchError
from .exceptions import ConstraintError


def equal_length(*args):
    '''Check that a tuple of arguments has the same length.
    '''
    for i, arg in enumerate(args):
        if not isinstance(arg, Sized):
            raise ExpectedTypeError(arg, ['Sized'])
        if i >= 1 and len(arg) != len(args[0]):
            raise DifferentLengthError(args, arg_tuple, arg)


@curry
def has_shape(shape, arg_id, args, kwargs):
    '''Check that a particular argument is an array with a given shape. The
    shape may contain string attributes, which will be fetched from arg0 to
    the function (usually self).
    '''
    self = args[0]
    arg = args[arg_id]
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


def is_sequence(arg_id, args, kwargs):
    arg = args[arg_id]
    if value != isinstance(arg, Iterable):
        raise ExpectedTypeError(arg, ['iterable'])


def is_float(arg_id, args, func_kwargs, **kwargs):
    arg = args[arg_id]
    if not isinstance(arg, float):
        raise ExpectedTypeError(arg, ['float'])
    if 'min' in kwargs and arg < kwargs['min']:
        raise ValueError("%s < min %s" % (arg, kwargs['min']))
    if 'max' in kwargs and arg > kwargs['max']:
        raise ValueError("%s > max %s" % (arg, kwargs['min']))


def is_array(arg_id, args, func_kwargs, **kwargs):
    arg = args[arg_id]
    if not isinstance(arg, ndarray):
        raise ExpectedTypeError(arg, ['ndarray'])


def operator_is_defined(op):
    @wrapt.decorator
    def checker(wrapped, instance, args, kwargs):
        print(instance)
        print('args', args, kwargs)
        if args[0] is None:
            raise ExpectedTypeError(instance, ['Model'])
        if op not in args[0]._operators:
            raise UndefinedOperatorError(op, instance, args[0], args[0]._operators)
        else:
            return wrapped(*args, **kwargs)
    return checker


def arg(arg_id, *constraints):
    @wrapt.decorator
    def arg_check_adder(wrapped, instance, args, kwargs):
        for check in constraints:
            check(arg_id, args, kwargs)
    return arg_check_adder


def args(*constraints):
    @wrapt.decorator
    def arg_check_adder(wrapped, instance, args, kwargs):
        for check in constraints:
            check(args)
    return arg_check_adder


def _resolve_names(args_by_name, method_args, *_, **kwargs):
    has_self = 'self' in method_args
    name2i = {name: i-has_self for i, name in enumerate(method_args)}
    constraints = [(name2i[n], s) for n, s in args_by_name.items()]
    return constraints
