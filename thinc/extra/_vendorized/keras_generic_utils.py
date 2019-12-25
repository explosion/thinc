# https://raw.githubusercontent.com/fchollet/keras/master/keras/utils/data_utils.py
# Copyright Francois Chollet, Google, others (2015)
# Under MIT license
import numpy as np
import time
import sys
import marshal
import types as python_types


def get_from_module(
    identifier, module_params, module_name, instantiate=False, kwargs=None
):
    if isinstance(identifier, str):
        res = module_params.get(identifier)
        if not res:
            raise ValueError("Invalid " + str(module_name) + ": " + str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif isinstance(identifier, dict):
        name = identifier.pop("name")
        res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise ValueError("Invalid " + str(module_name) + ": " + str(identifier))
    return identifier


def make_tuple(*args):
    return args


def func_dump(func):
    """Serialize user defined function."""
    code = marshal.dumps(func.__code__).decode("raw_unicode_escape")
    defaults = func.__defaults__
    if func.__closure__:
        closure = tuple(c.cell_contents for c in func.__closure__)
    else:
        closure = None
    return code, defaults, closure


def func_load(code, defaults=None, closure=None, globs=None):
    """Deserialize user defined function."""
    if isinstance(code, (tuple, list)):  # unpack previous dump
        code, defaults, closure = code
    code = marshal.loads(code.encode("raw_unicode_escape"))
    if globs is None:
        globs = globals()
    return python_types.FunctionType(
        code, globs, name=code.co_name, argdefs=defaults, closure=closure
    )


class Progbar(object):
    def __init__(self, target, width=30, verbose=1, interval=0.01):
        """Dislays a progress bar.

        # Arguments:
            target: Total number of steps expected.
            interval: Minimum visual progress update interval (in seconds).
        """
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], force=False):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [
                    v * (current - self.seen_so_far),
                    current - self.seen_so_far,
                ]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += current - self.seen_so_far
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")
            if self.target == -1:
                numdigits = 0
            else:
                numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = "%%%dd/%%%dd [" % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += "=" * (prog_width - 1)
                if current < self.target:
                    bar += ">"
                else:
                    bar += "="
            bar += "." * (self.width - prog_width)
            bar += "]"
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ""
            if current < self.target:
                info += " - ETA: %ds " % eta
            else:
                info += " - %ds" % (now - self.start)
            for k in self.unique_values:
                info += " - %s:" % k
                if isinstance(self.sum_values[k], list):
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                else:
                    info += " %s" % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += (prev_total_width - self.total_width) * " "

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = "%ds" % (now - self.start)
                for k in self.unique_values:
                    info += " - %s:" % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


def display_table(rows, positions):
    def display_row(objects, positions):
        line = ""
        for i in range(len(objects)):
            line += str(objects[i])
            line = line[: positions[i]]
            line += " " * (positions[i] - len(line))
        print(line)

    for objects in rows:
        display_row(objects, positions)
