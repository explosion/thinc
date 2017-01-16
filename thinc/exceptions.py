# coding: utf-8
from __future__ import unicode_literals

import traceback
from termcolor import colored as color


def get_error(title, *args, **kwargs):
    template = '\n\n\t{title}{info}{tb}\n'
    info = '\n'.join(['\t' + l for l in args]) if args else ''
    highlight = kwargs['highlight'] if 'highlight' in kwargs else False
    tb = _get_traceback(kwargs['tb'], highlight) if 'tb' in kwargs else ''
    return template.format(title=color(title, 'red', attrs=['bold']),
                           info=info, tb=tb).encode('utf8')


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
    spacing = '   '*i + color('>>>', 'red')
    if highlight:
        text = text.replace(str(highlight), color(str(highlight), 'yellow'))
    return template.format(sp=spacing, t=text)


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
    max_to_print_of_value = 200
    def __init__(self, bad_type, expected):
        self.tb = traceback.extract_stack()
        bad_type = repr(bad_type)
        if len(bad_type) >= self.max_to_print_of_value:
            half = int(self.max_to_print_of_value/2)
            bad_type = bad_type[:half] + ' ... ' + bad_type[-half:]
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
