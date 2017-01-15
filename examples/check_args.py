from __future__ import print_function
from collections import defaultdict
from thinc import check
# from thinc.check import ArgCheckAdder
# from thinc.exceptions import ExpectedIntError


def check_arg(arg_id, *constraints):
    return ArgCheckAdder(arg_id, *constraints)


def check_int_value(min=None, max=None):
    def constraint(arg_id, args, kwargs):
        if not isinstance(args[arg_id], int):
            raise ExpectedIntError(args[arg_id])
        assert args[arg_id] >= 0
    return constraint


#@check_arg(0, check_int_value(min=0))
#@check_arg(1, check_int_value(min=0))
#@check.args_equal_length((0, 1))
#@check.arg_is_sequence(0, 1)
@check.arg_is_float(0, 1)
def add_stuff(int1, int2):
    return int1 + int2


def main():
    print(add_stuff(15, 12))
    #print(add_stuff((1, 2), (0, 1, 2)))
    #print(add_stuff(["abc", "bye"], [1, 2]))
    #print(add_positive_integers(0, 5))
    #print(add_positive_integers(-1, 5))


if __name__ == '__main__':
    main()
