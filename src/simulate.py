#!/usr/bin/env python3
"""Analyze optical systems

This is vaguely like a ray tracer, but:
- Elements are encountered in a prescribed order.
- We track phase information (and maybe more stuff later).
- We trace from origin to a detector.

For consistency, distances are measured in meters throughout -- even wavelengths.
(Exception: Coefficients for the Sellmeier equation use squared microns because that's what
you'll find in tables.)

Instruments generally point in direction of positive z-axis.  We use right-handed coordinates.

We use homogeneous coordinates (x,y,z,w) and the usual way of using 4x4 matrices for transformations
(as in OpenGL, for instance).  There might be places where we do some tricks that interpret this as
the quaternion w + x i + y j + z k.
"""
# TODO: track what happens to different wavelengths.
# TODO: Handle polarization information.  Is this just going to
# be some matrix?
from __future__ import print_function, division # in case used from python2

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#import sys

from common import *
import geometry as geo
import instrument

def plot_segments(ax, points, pairs):
    use_lines = False
    use_lines = True
    if use_lines:
        segments = [(points[i], points[j]) for i,j in pairs]
        lc = LineCollection(segments)
        ax.add_collection(lc)
    else:
        # I would have thought that np.array(iterator) yields an array,
        # but when I pass that to plt.scatter it complains that it got an iterator.
        x = np.array([point[0] for point in points])
        y = np.array([point[1] for point in points])
        plt.scatter(x, y)

# Nots: On classical cassegrain, caught1 has +/- 1e-5 in y direction, similar in x direction but coma-like
def test0(do_ray_bundles=True):
    #setback = 5 * mm # to see how it affects focus
    setback = 0 * mm
    # setback = -0.2 + 1.99826723e-01 # closest to raybundle 1's focus.
    # instr, focal_length = instrument.newtonian_example()
    # instr, focal_length = instrument.classical_cassegrain_example()
    instr, focal_length = instrument.ritchey_chretien_example()
    # instr, focal_length = instrument.maksutov_example()
    # instr, focal_length = instrument.simple_refractor_example()

    instr = instr.setback_sensor(setback)

    R0 = np.eye(4) # head on
    caught0, pairs0 = instr.simulate(R0)

    # Now, rotate slightly around y-axis.
    # Note that a 35mm sensor is 24mm x 36mm.  So, let us consider an angle that puts us roughly
    # 8mm off the center of the sensor and then 16mm off the center.
    theta = 8 * mm / focal_length
    R1 = geo.make_rotation_y(theta)
    caught1, pairs1 = instr.simulate(R1)

    theta *= 2
    R2 = geo.make_rotation_y(theta)
    caught2, pairs2 = instr.simulate(R2)

    plot = False
    plot = True
    if plot:
        fig, ax = plt.subplots()
        # ax doesn't scale itself when we use LineCollection.
        # We could look at min/max x, y values, but we'll just make it
        # the size of our sensor.
        # We use a 36mm x 24mm sensor (this is basically 35mm film)
        ax.set_xlim(-18*mm,18*mm)
        ax.set_ylim(-12*mm,12*mm)
        # TODO: At a point where it's perfectly focused, we don't get a single pixel
        # from the line segments.
        plot_segments(ax, caught0, pairs0)
        plot_segments(ax, caught1, pairs1)
        plot_segments(ax, caught2, pairs2)
        plt.show()
    else:
        print(caught0)
        print(caught1)
        print(caught2)
    if do_ray_bundles:
        radius_scale = 0.1 # how much the source's radius gets scaled by when producing RayBundle
        x = []
        z = []
        approx_focal_lengths = []

        for bundle_index, R in enumerate([R0, R1, R2]):
            raybundle = instr.source.get_raybundle(R, radius_scale)
            raybundle_ = raybundle.cointeract(instr.elements)
            focus, info = raybundle_.approx_focus()
            x.append(focus[0])
            z.append(focus[2])
            approx_focal_length = raybundle_.approx_focal_length()
            approx_focal_lengths.append(approx_focal_length)
            print(f"raybundle {bundle_index} focus: {focus}, {info}; focal length {approx_focal_length}")
        x = np.array(x)
        z = np.array(z)
        plt.plot(x,z)
        plt.title("field curvature")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.show()

        V = np.vander(x ** 2)
        # We won't worry about singular matrices.
        coeffs = LA.solve(V, z)
        print(f"z = {coeffs[2]} + {coeffs[1]}x^2 + {coeffs[0]}x^4")




def test1():
    #setback = 5 * mm # to see how it affects focus
    setback = 0 * mm
    # instr, focal_length = instrument.newtonian_example()
    # instr, focal_length = instrument.classical_cassegrain_example()
    # instr, focal_length = instrument.ritchey_chretien_example()
    instr, focal_length = instrument.simple_refractor_example()

    instr = instrument.setback_sensor(setback)

    R0 = np.eye(4) # head on
    caught0, pairs0 = instr.simulate(R0)
    plot = True

    if plot:
        fig, ax = plt.subplots()
        # ax doesn't scale itself when we use LineCollection.
        # We could look at min/max x, y values, but we'll just make it
        # the size of our sensor.
        # We use a 36mm x 24mm sensor (this is basically 35mm film)
        ax.set_xlim(-18*mm,18*mm)
        ax.set_ylim(-12*mm,12*mm)
        # TODO: At a point where it's perfectly focused, we don't get a single pixel
        # from the line segments.
        plot_segments(ax, caught0, pairs0)
        plt.show()
    else:
        print(caught0)

if __name__ == "__main__":
    test0()
