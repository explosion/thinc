# coding: utf-8
from __future__ import unicode_literals

import traceback


def get_error(title, *args, **kwargs):
    template = '\n\n\t\033[91m\033[1m{title}\033[0m{info}{tb}\n'
    info = '\n' + '\n'.join(['\t' + arg for arg in args]) if args else ''
    highlight = kwargs['highlight'] if 'highlight' in kwargs else False
    tb = _get_traceback(kwargs['tb'], highlight) if 'tb' in kwargs else ''
    return template.format(title=title, info=info, tb=tb).encode('utf8')


def _get_traceback(tb, highlight):
    template = '\n\n\t\033[94m\033[1m{title}:\033[0m\n\t{tb}'
    tb_range = tb[-5:-2]
    tb_list = [_format_traceback(p, l, fn, t, i, len(tb_range), highlight) for i, (p, l, fn, t) in enumerate(tb_range)]
    tb_str = '\n'.join(tb_list).strip()
    return template.format(title='Traceback', tb=tb_str)


def _format_traceback(path, line, fn, text, i, count, highlight):
    template = '\t{i} \033[1m{fn}\033[0m [{l}] in \033[4m{p}\033[0m{t}'
    indent = ('└─' if i == count-1 else '├─') + '──'*i
    filename = path.rsplit('/thinc/', 1)[1] if '/thinc/' in path else path
    text = _format_user_error(text, i, highlight) if i == count-1 else ''
    return template.format(l=str(line), fn=fn, p=filename, i=indent, t=text)


def _format_user_error(text, i, highlight):
    template = '\n\t{sp} \033[91m>>>\033[0m {t}'
    if highlight:
        template_h = '\033[93m{t}\033[0m'.format(t=str(highlight))
        text = text.replace(str(highlight), template_h)
    return template.format(sp='   '*i, t=text)


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


class DifferentLengthError(ValueError):
    def __init__(self, args, arg_tuple, arg_id):
        self.tb = traceback.extract_stack()
        vals = ['{v} [{l}]'.format(v=args[arg_id], l=len(args[arg_id])) for arg_id in arg_tuple]
        ValueError.__init__(self, get_error(
            "Values need to be equal length: {v}".format(v=', '.join(vals)),
            tb=self.tb
        ))


class ShapeMismatchError(ValueError):
    def __init__(self, shape, dim, shape_names):
        self.tb = traceback.extract_stack()
        ValueError.__init__(self, get_error(
            "Shape mismatch: {s} does not have the right dimension {d}.".format(s=shape, d=dim),
            tb=self.tb
        ))


class ExpectedTypeError(TypeError):
    def __init__(self, bad_type, expected):
        self.tb = traceback.extract_stack()
        TypeError.__init__(self, get_error(
            "Expected type {e}, but got: {v} ({t})".format(e='/'.join(expected), v=bad_type, t=type(bad_type)),
            tb=self.tb,
            highlight=bad_type
        ))


class ConstraintError(ValueError):
    def __init__(self, bad_con, expected):
        self.tb = traceback.extract_stack()
        ValueError.__init__(self, get_error(
            "Invalid argument constraint: {v} ({t})".format(v=bad_con, t=type(bad_con)),
            "Expected {e}.".format(e=', '.join(expected)),
            tb=self.tb,
            highlight=bad_con
        ))
