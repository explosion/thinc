from __future__ import print_function


@check.arg(0, check.int_value(min=0))
@check.arg(1, check.int_value(min=0))
def add_positive_integers(int1, int2):
    return int1 + int2


def main():
    print(add_positive_integers(1, 5))
    print(add_positive_integers(0, 5))
    print(add_positive_integers(-1, 5))


if __name__ == '__main__':
    main()
