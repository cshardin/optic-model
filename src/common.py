import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

nm = 1e-9

blue = 450 * nm
green = 550 * nm
red = 650 * nm
