# coding: utf-8
from __future__ import unicode_literals

import traceback


def get_error(title, *args, **kwargs):
    template = '\n\n\t\033[91m\033[1m{title}\033[0m\n{info}{tb}\n'
    info = '\n'.join(['\t' + arg for arg in args])
    tb = get_traceback(kwargs['tb']) if 'tb' in kwargs else ''
    return template.format(title=title, info=info, tb=tb)


def get_traceback(tb):
    template = '\n\n\t\033[94m\033[1m{title}:\033[0m\n\t{tb}'
    tb_range = tb[-5:-2]
    tb_list = [_format_traceback(p, l, fn, i, len(tb_range)) for i, (p, l, fn, _) in enumerate(tb_range)]
    tb_str = '\n'.join(tb_list).strip()
    return template.format(title='Traceback', tb=tb_str)


def _format_traceback(path, line, fn, i, count):
    template = '\t{i} \033[1m{fn}\033[0m [{l}] in \033[4m{p}\033[0m'
    indent = ('└─' if i == count - 1 else '├─') + '──' * i
    filename = path.rsplit('/thinc/', 1)[1] if '/thinc/' in path else path
    return template.format(l=str(line), fn=fn, p=filename, i=indent)


class UndefinedOperatorError(TypeError):
    def __init__(self, op, arg1, arg2, operators):
        self.tb = traceback.extract_stack()

        msg = get_error(
            "Undefined operator: {op}".format(op=op),
            "Called by ({arg1}, {arg2})".format(arg1=arg1, arg2=arg2),
            "Available: {operators}".format(operators=operators),
            tb=self.tb
        )

        TypeError.__init__(self, msg.encode('utf8'))




# SHAPE_ERR_TEMPLATE = '''

# In the context of {context}:\n

# Shape1 != Shape2

# Where:

# Shape1={shape1}
# Shape2={shape2}
# '''

# class ShapeError(ValueError):
#     def __init__(self, shape1, shape2, context):
#         msg = SHAPE_ERR_TEMPLATE.strip().format(
#             shape1=shape1, shape2=shape2, context=context)
#         ValueError.__init__(self, msg)
#         self.tb = sys.exc_info()[2]

#     @classmethod
#     def dimensions_mismatch(cls, shape1, shape2, context):
#         if shape1 == shape2:
#             return None
#         else:
#             return cls(shape1, shape2, context)

#     @classmethod
#     def dim_mismatch(cls, expected, observed):
#         return cls("Dimension mismatch: %s vs %s" % (expected, observed))
