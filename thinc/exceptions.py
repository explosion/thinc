# coding: utf-8
from __future__ import unicode_literals

import traceback
from wasabi import TracebackPrinter, format_repr


get_error = TracebackPrinter(tb_base="thinc", tb_exclude=("check.py",))


class UndefinedOperatorError(TypeError):
    def __init__(self, op, arg1, arg2, operators):
        self.tb = traceback.extract_stack()
        TypeError.__init__(
            self,
            get_error(
                "Undefined operator: {op}".format(op=op),
                "Called by ({arg1}, {arg2})".format(arg1=arg1, arg2=arg2),
                "Available: {ops}".format(ops=", ".join(operators.keys())),
                tb=self.tb,
                highlight=op,
            ),
        )


class OutsideRangeError(ValueError):
    def __init__(self, arg, val, operator):
        self.tb = traceback.extract_stack()
        ValueError.__init__(
            self,
            get_error(
                "Outside range: {v} needs to be {o} {v2}".format(
                    v=format_repr(arg), o=operator, v2=format_repr(val)
                ),
                tb=self.tb,
            ),
        )


class DifferentLengthError(ValueError):
    def __init__(self, lengths, arg):
        self.tb = traceback.extract_stack()
        ValueError.__init__(
            self,
            get_error(
                "Values need to be equal length: {v}".format(v=format_repr(lengths)),
                tb=self.tb,
            ),
        )


class ShapeMismatchError(ValueError):
    def __init__(self, shape, dim, shape_names):
        self.tb = traceback.extract_stack()
        shape = format_repr(shape)
        dim = format_repr(dim)
        ValueError.__init__(
            self,
            get_error(
                "Shape mismatch: input {s} not compatible with {d}.".format(
                    s=shape, d=dim
                ),
                tb=self.tb,
            ),
        )


class TooFewDimensionsError(ValueError):
    def __init__(self, shape, axis):
        self.tb = traceback.extract_stack()
        ValueError.__init__(
            self,
            get_error(
                "Shape mismatch: input {s} has too short for axis {d}.".format(
                    s=format_repr(shape), d=axis
                ),
                tb=self.tb,
            ),
        )


class ExpectedTypeError(TypeError):
    max_to_print_of_value = 200

    def __init__(self, bad_type, expected):
        if isinstance(expected, str):
            expected = [expected]
        self.tb = traceback.extract_stack()
        TypeError.__init__(
            self,
            get_error(
                "Expected type {e}, but got: {v} ({t})".format(
                    e="/".join(expected), v=format_repr(bad_type), t=type(bad_type)
                ),
                tb=self.tb,
                highlight=format_repr(bad_type),
            ),
        )
