#!/usr/bin/env python3

import numpy as np
from geometry import reflect, refract, Ray, Quadric, Plane, translation3f, make_bound_vector, point, vector
from common import *
from material import *

# TODO: Add fraction of light that is absorbed/transmitted.  For simplicity, that
# will just be a constant; for now, that constant is implicitly 1 everywhere.
class SubElement:
    """A SubElement might be a reflector, or *one* side of a lens.

    A SubElement knows what material *comes next*.  So, the front of a lens
    is an element whose associated material is glass, while the back of a lens
    is an element whose associated material is air.
    """
    def __init__(self, geometry, clip=None, material=None):
        """
        Args:
            geometry: the actual geometry of the surface (e.g., the sphere it's a part of)
            clip: something with an `inside` method that can be used to test whether a
             proposed intersection point actually hits this object.  (A ray that does
             not hit is lost--it is assumed to be absorbed by something or to have
             left the instrument.)
            material: what material the element is made of (default: reflector)
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
            wavelength = ray.wavelength
            ior = self.material.get_ior(wavelength)
            new_v = refract(v, grad, ior)
        assert new_v[3] == 0., f"bad reflection/refraction {new_v} (not vector-like); v={v}; grad={grad}"
        new_v_q = np.stack([new_v, new_q], axis=1)
        return ray.with_v_q_and_phase(new_v_q, phase)

class Compound:
    """Just a bunch of elements sequentially"""
    def __init__(self, elements, comment=""):
        self.elements = elements
        self.comment = comment

    def interact(self, ray):
        debug = True
        if debug:
            print(f"compound interact ({self.comment}) got ray {ray.v_q}")
        for index, elt in enumerate(self.elements):
            ray = elt.interact(ray)
            if debug:
                if ray is not None:
                    print(f"{self.comment}{index}: {ray.v_q}")
                else:
                    print(f"{self.comment}{index}: {ray}")
            if ray is None:
                return None
        return ray

# You might think make_paraboloid and make_hyperboloid should class functions of Quadric, but
# note that Quadric represents the entire conic and has no material.
# TODO: I think make_conic does the right thing for all conics, not just hyperboloids.  Verify.
# TODO: Our convention for using normals to determine which surface you hit is kind of annoying...
def make_conic(R, K, z_offset, material=None, reverse_normal=False):
    """
    See https://en.wikipedia.org/wiki/Conic_constant

    r^2 - 2Rz + (K+1)z^2 = 0

    Be careful about the sign convention for radius of curvature.  We follow the convention
    in https://en.wikipedia.org/wiki/Conic_constant but this is opposite the convention in
    https://en.wikipedia.org/wiki/Lens#Lensmaker's_equation .

    Args:
        R: radius of curvature; use R > 0 for concave "up" (direction of positive z-axis) while R < 0
         is concave "down"
        K: conic constant; should be < -1 for hyperboloids, -1 for paraboloids, > -1 for ellipses
         (including 0 for spheres).  The relationship with eccentricity e is K = -e^2 (when
         K <= 0).
        z_offset: z-coordinate where the surface intersects z-axis
        material: mostly self explanatory; None means reflector
        reverse_normal: If true, surface points in direction of negative z-axis rather than
         positive z-axis
    """
    M = np.diag([1, 1, (K+1), 0])
    M[2,3] = -R
    M[3,2] = M[2,3]
    # For either sign of R, we want the convention that gradient points up at origin.
    # That gradient is (0,0,-R).
    # When R < 0, we already have that.
    # For R > 0, we need to negate M to get that.
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
    return SubElement(geometry, clip, material=material)

def make_lens(R1, R2, d, z_offset, material=None, external_material=None):
    """
    The resulting lens faces in direction of positive z-axis.

    The sign convention here is as in https://en.wikipedia.org/wiki/Lens#Lensmaker's_equation .
    That is:
    - R1 is curvature of lens closer to source (larger z).
    - R > 0 means center of curvature is at more negative z (farther along in path of light).
    - So for a common convex lens, R1 > 0 and R2 < 0.
    Careful: This is the opposite as above.

    The z-axis intersects first surface at z = z_offset + d/2 and the 2nd surface at z = z_offset - d/2.

    The default material is BK7.

    The lensmaker's equation:
    1/f = (n - 1)*[1/R1 - 1/R2 + (n-1)*d/(n * R1 * R2)]
    (n = ior)
    """
    # We handle defaults this way so that if None is passed we can fill in the default.
    if material is None:
        material = bk7
    if external_material is None:
        external_material = air
    center1 = z_offset + d/2 - R1
    center2 = z_offset - d/2 - R2
    sphere1 = Quadric.sphere(R1).untransform(translation3f(0,0,-center1))
    sphere2 = Quadric.sphere(R2).untransform(translation3f(0,0,-center2))
    # TODO: worry where the two surfaces meet and introduce appropriate clipping
    element1 = SubElement(sphere1, None, material=material)
    element2 = SubElement(sphere2, None, material=external_material)
    return Compound([element1, element2], comment="lens")

def get_focal_length(R1, R2, d, n):
    one_over_f = (n - 1) * (1/R1 - 1/R2 + (n-1)*d/(n * R1 * R2))
    return 1 / one_over_f

def make_lens_basic(f, d, z_offset):
    """Assume BK7 and green light, with R2 = -R1.

    We have
    1/f = (n - 1)*[1/R1 - 1/R2 + (n-1)*d/(n * R1 * R2)]
     = (n - 1) * [2/R1 - (n-1)*d/(n * R1^2)]
    R1^2 = f * (n - 1) * [2*R1 - (n-1)*d / n]
    R1^2 - f * (n - 1) * 2 * R1 + f d (n-1)^2/n = 0

    R1 = [2 f (n - 1) +- sqrt(...)]/2

    It seems weird that we'd get two values of R1 leading to same focal length.  Anyway, the one
    with -sqrt comes out very small in the examples I tried, so I guess would correspond to the light
    crossing over the center axis inside the lense.
    """
    material = bk7
    wavelength = green
    n = material.get_ior(wavelength)
    a = 1.
    b = -f * (n - 1) * 2
    c = f * d * (n - 1) ** 2 / n
    R1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    #R1_ = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    implied_f = get_focal_length(R1, -R1, d, n)
    #implied_f_ = get_focal_length(R1_, -R1_, d, n)
    print(f"R1={R1} yielded f={implied_f}")
    #print(f"R1={R1_} yielded f={implied_f_}")
    R2 = -R1
    return make_lens(R1, R2, d, z_offset, material=material, external_material=vacuum)

def test_lens_basic():
    f = 1. # 1000mm
    d = 0.01
    z_offset = 0
    make_lens_basic(f, d, z_offset)
