# coding: utf-8
from __future__ import unicode_literals

import traceback


def get_error(title, *args, **kwargs):
    template = '\n\n\t\033[91m\033[1m{title}\033[0m{info}{tb}\n'
    info = '\n' + '\n'.join(['\t' + arg for arg in args]) if args else ''
    tb = _get_traceback(kwargs['tb']) if 'tb' in kwargs else ''
    return template.format(title=title, info=info, tb=tb).encode('utf8')


def _get_traceback(tb):
    template = '\n\n\t\033[94m\033[1m{title}:\033[0m\n\t{tb}'
    tb_range = tb[-5:-2]
    tb_list = [_format_traceback(p, l, fn, t, i, len(tb_range)) for i, (p, l, fn, t) in enumerate(tb_range)]
    tb_str = '\n'.join(tb_list).strip()
    return template.format(title='Traceback', tb=tb_str)


def _format_traceback(path, line, fn, text, i, count):
    template = '\t{i} \033[1m{fn}\033[0m [{l}] in \033[4m{p}\033[0m{t}'
    template_text = '\n\t{sp} \033[91m>>>\033[0m {t}'
    indent = ('└─' if i == count-1 else '├─') + '──'*i
    filename = path.rsplit('/thinc/', 1)[1] if '/thinc/' in path else path
    text = template_text.format(sp='   '*i, t=text) if i == count-1 else ''
    return template.format(l=str(line), fn=fn, p=filename, i=indent, t=text)


class UndefinedOperatorError(TypeError):
    def __init__(self, op, arg1, arg2, operators):
        self.tb = traceback.extract_stack()
        msg = get_error(
            "Undefined operator: {op}".format(op=op),
            "Called by ({arg1}, {arg2})".format(arg1=arg1, arg2=arg2),
            "Available: {ops}".format(ops= ', '.join(operators.keys())),
            tb=self.tb
        )

        TypeError.__init__(self, msg)


class ExpectedIntError(TypeError):
    def __init__(self, no_int):
        self.tb = traceback.extract_stack()
        msg = get_error(
            "Expected an integer, but got: {no_int}".format(no_int=no_int),
            tb=self.tb
        )

        TypeError.__init__(self, msg)
