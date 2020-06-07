#!/usr/bin/env python3

import numpy as np
from geometry import reflect, refract, Ray, Quadric, Plane, translation3f, make_bound_vector, point, vector
from material import *

nm = 1e-9

green = 550 * nm

# TODO: Add fraction of light that is absorbed/transmitted.  For simplicity, that
# will just be a constant; for now, that constant is implicitly 1 everywhere.
# TODO: How should coming out the back of a lens work?  In particular, how does
# it know what the IOR of the outside space is (unless we assume some
# baseline)?  Should we actually think of it as not exiting a solid, but as
# entering a different solid whose ior is the ior of the air?  That might seem silly
# physically but I think it works well mathematically (although it means that
# lenses have a direction--weird in general but not weird in the way we're modeling
# these optical systems as having a prescribed order in which you hit stuff).
# It will also work well for systems where two lenses are flush with each other
# so you're exiting one element just as you're entering another; thinking of these
# two lenses as being comprised of 3 SubElements (the front of the first lens,
# the interface between the two lenses, and the back of the second lens) works
# well; what would the "inside" of that second SubElement be otherwise?
class SubElement():
    """A SubElement might be a reflector, or *one* side of a lens."""
    def __init__(self, geometry, clip=None, material=None):
        """
        Args:
            geometry: the actual geometry of the surface (e.g., the sphere it's a part of)
            clip: something with an `inside` method that can be used to test whether a
             proposed intersection point actually hits this object.
            ior: index of refraction (None for a reflector)
        """
        self.geometry = geometry
        self.clip = clip
        if material is None:
            material = perfect_mirror
        self.material = material

    def intersect(self, ray):
        """Whereas geometry.intersect returns just t, we return (t, q)"""
        # TODO: intersect in geometry needs to be refactored either to take in `clip`
        # or to return all intersections.  For now, we can rely on the normal as a disposer
        # that results in at most one intersection.
        t = self.geometry.intersect(ray)
        if t is None:
            return None
        R = ray.v_q
        q = R[:,0] * t + R[:,1]
        if self.clip is None or self.clip.inside(q):
            return (t, q)
        else:
            return None

    def interact(self, ray):
        """Compute reflected/transmitted ray"""
        tq = self.intersect(ray)
        if tq is None:
            return None
        (t, new_q) = tq
        phase = ray.phase + t
        grad = self.geometry.grad(new_q)
        v = ray.v()

        if self.material.is_reflector:
            new_v = reflect(v, grad)
        else:
            # For now, we just do green photons.
            ior = self.material.get_ior(green)
            new_v = refract(v, grad, ior)
        assert new_v[3] == 0., f"bad reflection/refraction {new_v} (not vector-like); v={v}; grad={grad}"
        new_v_q = np.stack([new_v, new_q], axis=1)
        return Ray(new_v_q, phase, ray.annotations)

# TODO: Should make_paraboloid and make_hyperboloid be class functions of Quadric?  No, because
# Quadric represents the entire conic and has no material.
# TODO: Does the following just do the right thing for other conics, not just
# hyperboloids?  If so, change name to make_conic?
# TODO: Our convention for using normals to determine which surface you hit is kind of annoying...
def make_hyperboloid(R, K, z_offset, material=None, reverse_normal=False):
    """
    See https://en.wikipedia.org/wiki/Conic_constant

    r^2 - 2Rz + (K+1)z^2 = 0

    Note that in optics there seem to be some unfortunate inconsistencies about sign conventions
    for radius of curvature.  In some places, R > 0 is concave "up" while in some places R < 0
    is concave up.  In particular, https://en.wikipedia.org/wiki/Lens#Lensmaker's_equation has
    the reverse of our sign convention.

    Args:
        R: radius of curvature; use R > 0 for concave "up" (direction of positive z-axis) while R < 0
        is concave "down"
        K: conic constant, should be < -1 for hyperboloids
    """
    M = np.diag([1, 1, (K+1), 0])
    M[2,3] = -R
    M[3,2] = M[2,3]
    # For either sign of R, we want the convention that gradient points up at origin.
    # That gradient is (0,0,-R).
    # When R < 0, we already have that.
    # For R > 0, we need to negate M to get that
    if R > 0:
        M *= -1
    if reverse_normal:
        M *= -1
    quad = Quadric(M)
    geometry = quad.untransform(translation3f(0,0,-z_offset))
    if R > 0:
        # We want to keep the top sheet.
        # TODO: Let clip_z be halfway between the two foci.
        clip_z = z_offset - 1e-6
        clip = Plane(make_bound_vector(point(0, 0, clip_z), vector(0, 0, -1)))
    else:
        clip_z = z_offset + 1e-6
        clip = Plane(make_bound_vector(point(0, 0, clip_z), vector(0, 0, 1)))
    # TODO: emulate ocaml's ?foo:bar (i.e., don't override default if value is None)
    return SubElement(geometry, clip, material=material)
