# coding: utf8
from __future__ import unicode_literals

from collections import Sequence, Sized, Iterable, Callable
from numpy import ndarray

from .compat import integer_types
from .extra import wrapt
from .exceptions import UndefinedOperatorError, DifferentLengthError
from .exceptions import ExpectedTypeError, ShapeMismatchError
from .exceptions import OutsideRangeError


def is_docs(arg_id, args, kwargs):
    from spacy.tokens.doc import Doc

    docs = args[arg_id]
    if not isinstance(docs, Sequence):
        raise ExpectedTypeError(type(docs), ["Sequence"])
    if not isinstance(docs[0], Doc):
        raise ExpectedTypeError(type(docs[0]), ["spacy.tokens.doc.Doc"])


def equal_length(*args):
    """Check that arguments have the same length.
    """
    for i, arg in enumerate(args):
        if not isinstance(arg, Sized):
            raise ExpectedTypeError(arg, ["Sized"])
        if i >= 1 and len(arg) != len(args[0]):
            raise DifferentLengthError(args, arg)


def equal_axis(*args, **axis):
    """Check that elements have the same dimension on specified axis.
    """
    axis = axis.get("axis", -1)
    for i, arg in enumerate(args):
        if not isinstance(arg, ndarray):
            raise ExpectedTypeError(arg, ["ndarray"])
        if axis >= 0 and (axis + 1) < arg.shape[axis]:
            raise ShapeMismatchError(arg.shape[axis], axis, [])
        if i >= 1 and arg.shape[axis] != args[0].shape[axis]:
            lengths = [a.shape[axis] for a in args]
            raise DifferentLengthError(lengths, arg)


def has_shape(shape):
    """Check that a particular argument is an array with a given shape. The
    shape may contain string attributes, which will be fetched from arg0 to
    the function (usually self).
    """

    def has_shape_inner(arg_id, args, kwargs):
        self = args[0]
        arg = args[arg_id]
        if not hasattr(arg, "shape"):
            raise ExpectedTypeError(arg, ["array"])
        shape_values = []
        for dim in shape:
            if not isinstance(dim, integer_types):
                dim = getattr(self, dim, None)
            shape_values.append(dim)
        if len(shape) != len(arg.shape):
            raise ShapeMismatchError(arg.shape, tuple(shape_values), shape)
        for i, dim in enumerate(shape_values):
            # Allow underspecified dimensions
            if dim is not None and arg.shape[i] != dim:
                raise ShapeMismatchError(arg.shape, shape_values, shape)

    return has_shape_inner


def is_shape(arg_id, args, func_kwargs, **kwargs):
    arg = args[arg_id]
    if not isinstance(arg, Iterable):
        raise ExpectedTypeError(arg, ["iterable"])
    for value in arg:
        if not isinstance(value, integer_types) or value < 0:
            raise ExpectedTypeError(arg, ["valid shape (positive ints)"])


def is_sequence(arg_id, args, kwargs):
    arg = args[arg_id]
    if not isinstance(arg, Iterable) and not hasattr(arg, "__getitem__"):
        raise ExpectedTypeError(arg, ["iterable"])


def is_float(arg_id, args, func_kwargs, **kwargs):
    arg = args[arg_id]
    if not isinstance(arg, float):
        raise ExpectedTypeError(arg, ["float"])
    if "min" in kwargs and arg < kwargs["min"]:
        raise OutsideRangeError(arg, kwargs["min"], ">=")
    if "max" in kwargs and arg > kwargs["max"]:
        raise OutsideRangeError(arg, kwargs["max"], "<=")


def is_int(arg_id, args, func_kwargs, **kwargs):
    arg = args[arg_id]
    if not isinstance(arg, integer_types):
        raise ExpectedTypeError(arg, ["int"])
    if "min" in kwargs and arg < kwargs["min"]:
        raise OutsideRangeError(arg, kwargs["min"], ">=")
    if "max" in kwargs and arg > kwargs["max"]:
        raise OutsideRangeError(arg, kwargs["max"], "<=")


def is_array(arg_id, args, func_kwargs, **kwargs):
    arg = args[arg_id]
    if not isinstance(arg, ndarray):
        raise ExpectedTypeError(arg, ["ndarray"])


def is_int_array(arg_id, args, func_kwargs, **kwargs):
    arg = args[arg_id]
    if not isinstance(arg, ndarray) or "i" not in arg.dtype.kind:
        raise ExpectedTypeError(arg, ["ndarray[int]"])


def operator_is_defined(op):
    @wrapt.decorator
    def checker(wrapped, instance, args, kwargs):
        if instance is None:
            instance = args[0]
        if instance is None:
            raise ExpectedTypeError(instance, ["Model"])
        if op not in instance._operators:
            raise UndefinedOperatorError(op, instance, args[0], instance._operators)
        else:
            return wrapped(*args, **kwargs)

    return checker


def arg(arg_id, *constraints):
    @wrapt.decorator
    def checked_function(wrapped, instance, args, kwargs):
        # for partial functions or other C-compiled functions
        if not hasattr(wrapped, "checks"):  # pragma: no cover
            return wrapped(*args, **kwargs)
        if instance is not None:
            fix_args = [instance] + list(args)
        else:
            fix_args = list(args)
        for arg_id, checks in wrapped.checks.items():
            for check in checks:
                if not isinstance(check, Callable):
                    raise ExpectedTypeError(check, ["Callable"])
                check(arg_id, fix_args, kwargs)
        return wrapped(*args, **kwargs)

    def arg_check_adder(func):
        if hasattr(func, "checks"):
            func.checks.setdefault(arg_id, []).extend(constraints)
            return func
        else:
            wrapped = checked_function(func)
            wrapped.checks = {arg_id: list(constraints)}
            return wrapped

    return arg_check_adder


def args(*constraints):
    @wrapt.decorator
    def arg_check_adder(wrapped, instance, args, kwargs):
        for check in constraints:
            if not isinstance(check, Callable):
                raise ExpectedTypeError(check, ["Callable"])
            check(*args)
        return wrapped(*args, **kwargs)

    return arg_check_adder
