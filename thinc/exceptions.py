# coding: utf-8
from __future__ import unicode_literals
from collections import Sized

import os
import traceback


class UndefinedOperatorError(TypeError):
    def __init__(self, op, arg1, arg2, operators):
        self.tb = traceback.extract_stack()
        TypeError.__init__(self, get_error(
            "Undefined operator: {op}".format(op=op),
            "Called by ({arg1}, {arg2})".format(arg1=arg1, arg2=arg2),
            "Available: {ops}".format(ops= ', '.join(operators.keys())),
            tb=self.tb,
            highlight=op
        ))


class OutsideRangeError(ValueError):
    def __init__(self, arg, val, operator):
        self.tb = traceback.extract_stack()
        ValueError.__init__(self, get_error(
            "Outside range: {v} needs to be {o} {v2}".format(
                v=_repr(arg), o=operator, v2=_repr(val)),
            tb=self.tb
        ))


class DifferentLengthError(ValueError):
    def __init__(self, lengths, arg):
        self.tb = traceback.extract_stack()
        ValueError.__init__(self, get_error(
            "Values need to be equal length: {v}".format(v=_repr(lengths)),
            tb=self.tb
        ))


class ShapeMismatchError(ValueError):
    def __init__(self, shape, dim, shape_names):
        self.tb = traceback.extract_stack()
        shape = _repr(shape)
        dim = _repr(dim)
        ValueError.__init__(self, get_error(
            "Shape mismatch: input {s} not compatible with {d}.".format(s=shape, d=dim),
            tb=self.tb
        ))


class TooFewDimensionsError(ValueError):
    def __init__(self, shape, axis):
        self.tb = traceback.extract_stack()
        ValueError.__init__(self, get_error(
            "Shape mismatch: input {s} has too short for axis {d}.".format(
            s=_repr(shape), d=axis), tb=self.tb
        ))


class ExpectedTypeError(TypeError):
    max_to_print_of_value = 200
    def __init__(self, bad_type, expected):
        if isinstance(expected, str):
            expected = [expected]
        self.tb = traceback.extract_stack()
        TypeError.__init__(self, get_error(
            "Expected type {e}, but got: {v} ({t})".format(e='/'.join(expected), v=_repr(bad_type), t=type(bad_type)),
            tb=self.tb,
            highlight=_repr(bad_type)
        ))


def get_error(title, *args, **kwargs):
    template = '\n\n\t{title}{info}{tb}\n'
    info = '\n'.join(['\t' + l for l in args]) if args else ''
    highlight = kwargs['highlight'] if 'highlight' in kwargs else False
    tb = _get_traceback(kwargs['tb'], highlight) if 'tb' in kwargs else ''
    return template.format(title=color(title, 'red', attrs=['bold']),
                           info=info, tb=tb)

def _repr(obj, max_len=50):
    string = repr(obj)
    if len(string) >= max_len:
        half = int(max_len/2)
        return string[:half] + ' ... ' + string[-half:]
    else:
        return string


def _get_traceback(tb, highlight):
    template = '\n\n\t{title}\n\t{tb}'
    # Prune "check.py" from tb (hacky)
    tb = [record for record in tb if not record[0].endswith('check.py')]
    tb_range = tb[-5:-2]
    tb_list = [_format_traceback(p, l, fn, t, i, len(tb_range), highlight) for i, (p, l, fn, t) in enumerate(tb_range)]
    return template.format(title=color('Traceback:', 'blue', attrs=['bold']),
                           tb='\n'.join(tb_list).strip())


def _format_traceback(path, line, fn, text, i, count, highlight):
    template = '\t{i} {fn} [{l}] in {p}{t}'
    indent = ('└─' if i == count-1 else '├─') + '──'*i
    filename = path.rsplit('/thinc/', 1)[1] if '/thinc/' in path else path
    text = _format_user_error(text, i, highlight) if i == count-1 else ''
    return template.format(l=str(line), i=indent, t=text,
                           fn=color(fn, attrs=['bold']),
                           p=color(filename, attrs=['underline']))


def _format_user_error(text, i, highlight):
    template = '\n\t  {sp} {t}'
    spacing = '  '*i + color(' >>>', 'red')
    if highlight:
        text = text.replace(str(highlight), color(str(highlight), 'yellow'))
    return template.format(sp=spacing, t=text)


def color(text, fg=None, attrs=None):
    """Wrap text in color / style ANSI escape sequences."""
    if os.getenv('ANSI_COLORS_DISABLED') is not None:
        return text
    attrs = attrs or []
    tpl = '\x1b[{}m'
    styles = {'red': 31, 'blue': 34, 'yellow': 33, 'bold': 1, 'underline': 4}
    style = ''
    for attr in attrs:
        if attr in styles:
            style += tpl.format(styles[attr])
    if fg and fg in styles:
        style += tpl.format(styles[fg])
    return '{}{}\x1b[0m'.format(style, text)
