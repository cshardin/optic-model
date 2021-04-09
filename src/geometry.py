#!/usr/bin/env python3
"""
Basic geometry

The boundary between geometry and everything else is admittedly a little fuzzy.
"""

# TODO: Figure out what to do in doc comments about the following issue.
# It's standard in Python doc comments to *end* with a list of the arguments.
# But often I want to be able to refer to those when describing what the function
# does...

from __future__ import print_function, division # in case used from python2

import numpy as np
import numpy.linalg as LA

from common import *

# Some linear algebra utility stuff
# def make_axes(v):
#     """Make a right-handed axis system where last vector points in direction of
#     of v.  Returned as 4x4 matrix whose first 3 columns are the axes and last column
#     is (0,0,0,1).
#     """
#     # TODO: fix so that it works if last column has zero z coordinate.
#     M = np.zeros(4)
#     M[3,0] = 1.
#     M[:,1] = v
#     M[0,2] = 1.
#     M[1,3] = 1.
#     (Q,R) = LA.qr(M)
#     assert R[0,0] > 0, "unexpected sign"
#     Q = Q[:,::-1] # put (0,0,0,1) at the back, and (0,0,1,0) at Q[:,2].
#     ...

def point(x, y, z):
    return np.array([x,y,z,1.])

def vector(x, y, z):
    return np.array([x,y,z,0.])

def vector3v(xyz):
    result = np.zeros(4)
    result[:3] = xyz
    return result

def translation3f(x, y, z):
    result = np.eye(4)
    result[0,3] = x
    result[1,3] = y
    result[2,3] = z
    return result

ii = vector(1,0,0)
jj = vector(0,1,0)
kk = vector(0,0,1)
origin = point(0,0,0)

def cross(u, v):
    """Cross product.

    Formally, we define u x v = 0.5 (u v - v u) where u and v are interpreted as quaternions.
    However, you can verify that this just annihilates the scalar parts.
    """
    return vector3v(np.cross(u[:-1], v[:-1]))

def make_rotation_y(theta):
    """Rotation by theta radians about y-axis"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

# TODO: Maybe BoundVector should be called Ray, and what is currently
# Ray should become PhasedRay or something like that.
class BoundVector():
    """A bound vector: a point in R^3 paired with a direction.  I.e., an element of the
    tangent bundle of R^3"""
    # TODO: v_q is so ugly.
    def __init__(self, v_q):
        self.v_q = v_q

    def q(self):
        return self.v_q[:,1]

    def v(self):
        return self.v_q[:,0]

def make_bound_vector(q, v):
    v_q = np.stack([v, q], axis=1)
    return BoundVector(v_q)

# TODO: Pull a lot of what's in Ray into BoundVector and have Ray derive BoundVector.
class Ray():
    def __init__(self, v_q, phase, annotations):
        """q is starting position [x; y; z; 1], v is direction
        [dx; dy; dz; 0], r = v_q is [v, q], phase is a kind of "distance traveled
        so far" and more properly, time so far (so, phase increases
        more per meter in stuff with high refactive index).

        `annotations` is a list of objects with arbitrary meaning.  This list
        may get modified.

        I think length of v will be speed of light in current medium.
        (This will be used in refraction calculations.)

        TODO: Add variable for how much has been absorbed so far.  For now reflections/tranmissions
        are perfect.

        TODO: Add some higher-order information, namely, rather than a pointlike photon, we think
        of it as a distorted infinitesimal disk (or even ellipsoid?) of photons, and a matrix that tells us
        how the direction varies with displacement of starting point from nominal starting point (in
        coordinates relative to the disk, so even as the size of the disk shrinks to zero it's not
        necessary to make the matrix go to infinity).
        E.g., with parabolic reflector, that little bundle is spread out over a little circle, and
        that matrix is zero; after the on-axis reflection, it's still spread out over a little circle,
        but the directions converge; where it reaches the focus, the radius of the disk is zero but
        we have the same spread of directions.
        To handle reflections of these bundles, we need 2nd order information about the surface.
        Does this also need some kind of quadratic form that says how phase varies over the disk?
        (E.g., after bouncing off the parabola, the little disk would be bent--but we don't want
        to bend the disk; but we can project the photons back onto a flat disk and adjust
        their phases.)  Or, maybe all that isn't worth the complexity--you can just trace an
        actual bundle of photons to get an approximation of the same info.
        """
        assert v_q.shape == (4,2), f"bad v_q = {v_q}"
        self.v_q = v_q
        self.phase = phase
        self.annotations = annotations

    @classmethod
    def of_q_v(cls, q, v):
        v_q = np.stack([v, q], axis=1)
        return cls(v_q, 0., [])

    def transform(self, M):
        assert M.shape == (4,4), f"bad M = {M}"
        return self.__class__(M.dot(self.v_q), self.phase, self.annotations)

    def q(self):
        return self.v_q[:,1]

    def v(self):
        return self.v_q[:,0]

    def advance_time(self, dt):
        M = np.eye(2)
        M[0,1] = dt
        v_q = self.v_q.dot(M) # basically, add t*v to q
        return self.__class__(v_q, self.phase + dt, self.annotations)

def solve_quadratic(a, b, c, which):
    """Solve a quadratic a t^2 + b t + c = 0.

    For our purposes, double roots are problematic and we'll "pretend" that there is no
    root at all.

    which='neg' means we take the -b - sqrt(...) solution; this is not necessarily the
    lesser solution, since a might be negative
    """
    # TODO: When a is very small (which will happen a lot for us, because we'll have
    # slightly off-axis rays hitting parabolas), there's bad cancellation error and
    # we should polish.
    if a == 0:
        assert b != 0, "degenerate equation"
        return -c/b
    discriminant = b * b - 4 * a * c
    if discriminant > 0:
        # pick the point where normal points towards us?  Or something else?
        return (-b - np.sqrt(discriminant))/(2 * a)
    else:
        return None # Missed, or a grazing hit, or nan, which we'll count as missing.

# TODO: Perhaps OrientedSurface should go away, and reflect and interact should
# move to SubElement.

# Let's say that a geometric object only needs to know how to compute intersections
# and surface normals. Things like reflection or refraction should be separate
# or in a parent module.
class OrientedSurface():
    """An OrientedSurface is something for which you can:
    - Compute intersection with a ray
    - Get a surface normal at a point on the surface
    - Ask whether you are "inside" or not (this is the oriented part).
    By convention, the surface normal should point to the outside.

    Our mental model (for which there are no exceptions yet) is that such a surface
    is defined by g(q) = c where q maps points to scalars and c is a constant,
    `grad` will be gradient of g, and being inside will mean g(q) <= c.
    """
    def intersect(self, ray):
        raise NotImplementedError("child must implement intersect")

    def grad(self, point):
        """Return value of grad should not be modified"""
        raise NotImplementedError("child must implement grad")

    def inside(self, point):
        raise NotImplementedError("child must implement inside")

    def reflect_deprecated(self, ray):
        t = self.intersect(ray)
        if t is None:
            return None
        R = ray.v_q
        new_q = R[:,0] * t + R[:,1]
        phase = ray.phase + t
        grad = self.grad(new_q)
        v = R[:,0]
        v_dot_grad = v.dot(grad)
        assert v_dot_grad < 0, "thought gradient pointed towards us"
        # To reflect, we subtract twice the projection onto grad.
        p_v_onto_grad = (v_dot_grad / grad.dot(grad)) * grad
        reflected_v = v - 2 * p_v_onto_grad
        assert reflected_v[3] == 0., f"bad reflection {reflected_v} (not vector-like); v={v}; grad={grad}"
        new_v_q = np.stack([reflected_v, new_q], axis=1)
        return Ray(new_v_q, phase, ray.annotations)

    def interact_deprecated(self, ray):
        """Compute reflected/transmitted ray"""
        if self.ior is None:
            return self.reflect(ray)
        else:
            raise NotImplementedError()

# TODO: Figure out where ior should "live".  E.g., should a lens surface *be* a Quadric
# (or an instance of a subclass of Quadric), or should it be an object that *has* a
# Quadric as its geometry?  Note that we want to also be able to have a geometry like
# a Plane, say, with no duplication of the logic of reflecting or refracting.
# I think a physical object should be something that *has* an OrientedSurface,
# and has a second optional OrientedSurface as its "clipping" region, and has
# surface properties like (possibly) an ior.

# TODO: Refactor how we handle which of the 2 intersection points to keep.  Like,
# for hyperboloids of 2 sheets, keeping track of which sheet you want based on the
# direction of the surface normals is flaky--a ray in a weird place could be treated
# as intersecting the wrong sheet rather than missing.
class Quadric(OrientedSurface):
    def __init__(self, M):
        """Quadric surface defined by 0.5 v' M v = 0 where v is in
        homogeneous coordinates.  M should be symmetric.

        ior: index of refraction; None for reflectors

        For now, rather than specifying a bounding box, we just assume
        you hit the part where the normal vector is "facing" you.

        TODO: Right now it's always a reflector; allow it to be a lens.
        TODO: Have a bounding region; allow annotations.  (E.g., if a ray
        lands outside the physical dimensions of the mirror, let ourselves
        trace it anyway with an annotation that says it missed the element.
        So, we can still see what the image would be "in principle" and
        see what portions were chopped off.)
        """
        self.M = M

    def __str__(self):
        return f"{self.M}"

    def intersect(self, ray):
        """Position is [v,q] * [t;1]...

        Of the (typically) two points where we intersect, we take the one where gradient is pointing more "at"
        us than away.  (Think of the gradient as a surface normal, and this is a primitive form of hidden
        surface removal.)
        """
        R = ray.v_q
        #print(f"R.shape={R.shape}")
        # TODO: fix bug: R has shape (2,) but we expect (4,2).
        RtMR = np.einsum('ji,jk,kl->il', R, self.M, R)
        # This is now 2x2 [a, b/2; b/2, c] and represents a t^2 + b t + c.
        # Solutions are (-b pm sqrt(b^2 - 4 a c))/(2 a)
        # When a > 0, we're going "downhill" at the first intersection and
        # "uphill" at the 2nd, so would want first value; when a < 0 it's
        # the other way around and we'd want the second value, but since
        # denominator is negative, that's still the - case of the +/-.
        a = RtMR[0,0]
        b = 2 * RtMR[0,1]
        c = RtMR[1,1]
        t = solve_quadratic(a, b, c, which='neg_sqrt')
        return t

    def grad(self, q):
        result = self.M.dot(q)
        result[3] = 0
        return result

    def inside(self, q):
        return q.dot(self.M).dot(q) <= 0.

    def untransform(self, A):
        """Apply inverse of A to the quadric."""
        return Quadric(A.T.dot(self.M).dot(A))

    @staticmethod
    def sphere(radius):
        """Negative radius yields sphere with normal pointing inward"""
        M = np.diag([1., 1., 1., -(radius ** 2)])
        if radius < 0:
            M *= -1
        return Quadric(M)


class Plane(OrientedSurface):
    # TODO: Even though we have the same amount of information as a bound vector,
    # it's not obvious that this fact is useful (unless there's some basic operation
    # involving bound vectors that would be useful here that doesn't exist yet).
    # Maybe we should just store q0 and v0 as separate instance variables?
    def __init__(self, bound_vector):
        """To reduce roundoff error, a plane is slightly overparametrized
        by specifying a point q0 on the plane and a normal vector v0.

        The associated equation is g(q) = 0 where g(q) = (q - q0).dot(v0);
        in particular, grad g = v0.
        """
        self.bound_vector = bound_vector

    def intersect(self, ray):
        # If ray is q + t v and we are q0, v0, we need
        # (q + t v - q0).dot(v0) = 0
        # t v.dot(v0) = (q - q0).dot(v0)
        # t = (q - q0).dot(v0) / v.dot(v0)
        denominator = ray.v().dot(self.bound_vector.v())
        if denominator == 0:
            return None
        else:
            numerator = (ray.q() - self.bound_vector.q()).dot(self.bound_vector.v())
            return numerator / denominator

    def grad(self, _q):
        return self.bound_vector.v()

    def inside(self, q):
        return (q - self.bound_vector.q()).dot(self.bound_vector.v())

def reflect(v, grad):
    v_dot_grad = v.dot(grad)
    # assert v_dot_grad < 0, "thought gradient pointed towards us" --obsolete
    # To reflect, we subtract twice the projection onto grad.
    p_v_onto_grad = (v_dot_grad / grad.dot(grad)) * grad
    new_v = v - 2 * p_v_onto_grad
    return new_v


# TODO: "The outgoing velocity will have magnitude self.ior"  I think we are
# using ior in some places where we want 1/ior.  Recall that ior = 4 means 1/4 speed of light.
def refract(v, grad, ior):
    """
    Given point q on surface and velocity v of incoming ray, compute outgoing
    velocity.

    The outgoing velocity will have magnitude 1/ior.

    Args:
        v: ray's velocity at intersection point
        grad: surface normal, not necessarily normalized
    """
    debug=False
    # We follow notation of https://en.wikipedia.org/wiki/Snell%27s_law
    # Values of n1, n2 are ior, not velocities.
    n1 = 1 / LA.norm(v)
    n2 = ior
    r = n1 / n2
    ell = v * n1
    n = grad / LA.norm(grad)
    # Do we want to negate n if its dot product with ell is positive?
    # Or do we assume things are set up so that this doesn't happen?
    c = - (n.dot(ell))
    if c < 0:
        c = -c
        n = -n
    v_refract = r * ell + (r * c - np.sqrt(1 - r * r * (1 - c * c))) * n
    if debug:
        print(f"refract: l={ell}, n={n}, v_refract={v_refract}, returning {v_refract*n2}; n1={n1}, n2={n2}; ||v_refract||={LA.norm(v_refract)}")
    return v_refract / n2

