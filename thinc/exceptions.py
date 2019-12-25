import traceback
from wasabi import TracebackPrinter, format_repr


get_error = TracebackPrinter(tb_base="thinc", tb_exclude=("check.py",))


class UndefinedOperatorError(TypeError):
    def __init__(self, op, arg1, arg2, operators):
        self.tb = traceback.extract_stack()
        TypeError.__init__(
            self,
            get_error(
                f"Undefined operator: {op}",
                f"Called by ({arg1}, {arg2})",
                f"Available: {', '.join(operators.keys())}",
                tb=self.tb,
                highlight=op,
            ),
        )


class OutsideRangeError(ValueError):
    def __init__(self, arg, val, operator):
        self.tb = traceback.extract_stack()
        title = f"Outside range: {format_repr(arg)} needs to be {operator} {format_repr(val)}"
        ValueError.__init__(self, get_error(title, tb=self.tb))


class DifferentLengthError(ValueError):
    def __init__(self, lengths, arg):
        self.tb = traceback.extract_stack()
        title = f"Values need to be equal length: {format_repr(lengths)}"
        ValueError.__init__(self, get_error(title, tb=self.tb))


class ShapeMismatchError(ValueError):
    def __init__(self, shape, dim, shape_names):
        self.tb = traceback.extract_stack()
        shape = format_repr(shape)
        dim = format_repr(dim)
        title = f"Shape mismatch: input {shape} not compatible with {dim}."
        ValueError.__init__(self, get_error(title, tb=self.tb))


class TooFewDimensionsError(ValueError):
    def __init__(self, shape, axis):
        self.tb = traceback.extract_stack()
        title = f"Shape mismatch: input {shape} has too short for axis {axis}."
        ValueError.__init__(self, get_error(title, tb=self.tb))


class ExpectedTypeError(TypeError):
    max_to_print_of_value = 200

    def __init__(self, bad_type, expected):
        if isinstance(expected, str):
            expected = [expected]
        self.tb = traceback.extract_stack()
        title = f"Expected type {'/'.join(expected)}, but got: {format_repr(bad_type)} ({type(bad_type)})"
        TypeError.__init__(
            self, get_error(title, tb=self.tb, highlight=format_repr(bad_type))
        )
