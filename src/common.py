import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

inch = 0.0254
mm = 1e-3
cm = 1e-2
nm = 1e-9

blue = 450 * nm
green = 550 * nm
red = 650 * nm

