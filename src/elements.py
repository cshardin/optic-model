#!/usr/bin/env python3

from geometry import reflect, refract, Ray
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
