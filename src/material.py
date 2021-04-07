#!/usr/bin/env python3
"""
Materials that reflect or refract.

For now, these are idealized and reflect or transmit 100% of light.

A material should have boolean variable `is_reflector` which is True for reflectors
and False for refractors.

When `is_reflector` is False, the material should have method `get_ior` that returns
index of refraction (as a function of wavelength measured IN METERS).

We'll use blue = 450nm, green = 550nm, red = 650nm.  Where lenses are treated as
having a single focal length, the convention in this package is that it's the
focal length for green.
"""
import numpy as np

class Sellmeier():
    """For materials whose ior as function of wavelength is described by the Sellmeier
    equation."""
    def __init__(self, b, c):
        """Units for Sellmeier equation: B is unitless; for the C, units
        are squared microns (as that's what's in most tables).
        """
        self.b = b
        self.c = c
        self.is_reflector = False

    def get_ior(self, wavelength):
        """Wavelength is in meters"""
        # lambda^2, as um^2
        lambda2 = (wavelength * 1e6) ** 2
        n2 = 1 + lambda2 * (self.b / (lambda2 - self.c)).sum()
        return np.sqrt(n2)

class Simple_refractor():
    """Material with constant ior"""
    def __init__(self, ior):
        self.is_reflector = False
        self.ior = ior

    def get_ior(self, _wavelength):
        return self.ior


# TODO: Is the following absolute (i.e., for glass in a vacuum), or assuming air at
# a certain temperature and pressure?  I got these coefficients from
# https://en.wikipedia.org/wiki/Sellmeier_equation but that article doesn't say
# which convention is being used.
# borosilicate crown glass
bk7 = Sellmeier(np.array([1.03961212, 0.231792344, 1.01046945]),
                np.array([6.00069867e-3, 2.00179144e-2, 103.560653]))

# Air at 1 atm and 0C
# If we want to be more accurate, this does depend on wavelength.
# Could look at https://refractiveindex.info/
air = Simple_refractor(1.000293)

# This is boring for now because we don't have any other properties...
class Reflector():
    def __init__(self):
        self.is_reflector = True

perfect_mirror = Reflector()
