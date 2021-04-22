#!/usr/bin/env python3
"""
Basic geometry

The boundary between geometry and everything else is admittedly a little fuzzy.
"""

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

# TODO: Should BoundVector derive ndarray?  For some functions that logically take a
# BoundVector, I want to be able to pass what we are calling v_q.
# TODO: Maybe BoundVector should be called Ray, and what is currently
# Ray should become PhasedRay or something like that.
class BoundVector:
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

# You might think we should get rid of phase and just modify v_q.  However, even if that
# doesn't contribute meaningful cancellation error (I don't know if it does), it makes
# it harder to interpret debug output or to draw the path that a photon followed.
# TODO: Pull a lot of what's in Ray into BoundVector and have Ray derive BoundVector?
class Ray:
    def __init__(self, v_q, phase, wavelength=green, annotations=[]):
        """q is starting position [x; y; z; 1], v is direction
        [dx; dy; dz; 0], r = v_q is [v, q], phase is a kind of "distance traveled
        so far" and more properly, time so far (so, phase increases
        more per meter in stuff with high refactive index).

        `annotations` is a list of objects with arbitrary meaning.  This list
        may get modified.

        The norm of v is speed of light in current medium. (This is used in
        refraction calculations.)

        Note that phase is measured in units of time and doesn't care about wavelength, so
        it's not literally phase in a wave sense.  (Our units of time are a little
        weird and very short: 1 unit of time is how long it takes a particle with velocity 1 to go
        distance 1; for velocity, 1 means speed of light in air or vacuum (depends), and
        for distance 1 means 1 meter.)

        TODO: Add variable for how much has been absorbed so far?  For now reflections/tranmissions
        are perfect.

        """
        assert v_q.shape == (4,2), f"bad v_q = {v_q}"
        self.v_q = v_q
        self.phase = phase
        self.wavelength = wavelength
        self.annotations = annotations

    @classmethod
    def of_q_v(cls, q, v, wavelength=green):
        v_q = np.stack([v, q], axis=1)
        return cls(v_q, 0., wavelength, [])

    def transform(self, M):
        assert M.shape == (4,4), f"bad M = {M}"
        return self.__class__(M.dot(self.v_q), self.phase, self.wavelength, self.annotations)

    def q(self):
        return self.v_q[:,1]

    def v(self):
        return self.v_q[:,0]

    def zero_phase_v_q(self):
        """The v_q that we'd have if we extrapolated backward to phase 0."""
        M = np.eye(2)
        M[0,1] = -self.phase
        return self.v_q.dot(M)

    def advance_time(self, dt):
        M = np.eye(2)
        M[0,1] = dt
        v_q = self.v_q.dot(M) # basically, add t*v to q
        return self.__class__(v_q, self.phase + dt, self.wavelength, self.annotations)

    def with_v_q_and_phase(self, v_q, phase):
        return self.__class__(v_q, phase, self.wavelength, self.annotations)

class CircularSource:
    def __init__(self, T, radius):
        """A circular source; T is transformation matrix (that will be applied to rays starting near
        origin in xy-plane headed in direction of z axis)"""
        assert T.shape == (4,4), f"bad T = {T}"
        self.T = T
        self.radius = radius

    def get_rays(self, pre_perturb, num_circles=3, resolution=6):
        """Returns (rays, pairs).
        pre_perturb is perturbation applied to rays before they are transformed by self's
        transformation.
        """
        # Below, u,v,w will denote local right-handed axis system, with *w*
        # pointing in direction of self.v
        debug = True
        radius = self.radius
        M = self.T.dot(pre_perturb)
        assert M.shape == (4,4), f"bad shape T={self.T}, pre_perturb={pre_perturb}"
        rays = []
        pairs = []
        def emit(u,v):
            ray = Ray.of_q_v(point(u,v,0), kk).transform(M)
            if debug:
                print(f"emitting {ray.v_q}")
            rays.append(ray)
        for i in range(0, num_circles+1):
            rays_so_far = len(rays)
            r = radius * (i / num_circles)
            num_points = max(1, resolution * i)
            for theta in np.arange(num_points) * (2 * np.pi / num_points):
                emit(r * np.cos(theta), r * np.sin(theta))
            for i in range(num_points):
                pairs.append((rays_so_far + i, rays_so_far + (i + 1) % num_points))
        # Add a couple more line segments...
        if num_circles >= 1:
            pairs.append((0,1))
            # Let's say we have 1 point in center, 6 points around that, then 12 points
            # around that; we want the 4th point of those 12; so it should have 1 + 6 + (12 / 4)
            # points before it.
            if num_circles >= 2:
                pairs.append((0, 1 + resolution + (resolution // 2)))
        if debug:
            print(f"pairs={pairs}")
            print(f"first few ray points = {[ ray.q() for ray in rays[:3]]}")
        return rays, pairs

class LinearSource:
    def __init__(self, radius, z):
        """A simple source, not very configurable; always points down, and
        all rays have y=0.
        """
        self.radius = radius
        self.z = z

    def get_rays(self, pre_perturb):
        # TODO: obey pre_perturb.
        num_rays = 5 # Very low, since this is basically for debugging.
        xs = np.linspace(-self.radius, self.radius, num_rays)
        rays = [Ray.of_q_v(point(x, 0, self.z), -kk) for x in xs]
        for ray in rays:
            print(f"emitting ray {ray.v_q}")
        pairs = [(k,k+1) for k in range(num_rays-1)]
        return rays, pairs


class RayBundle:
    """A RayBundle is like a Ray with additional local information about how certain perturbations
    of that ray would be affected.

    Rather than doing any calculus, we'll just implement this as multiple rays.

    For now, we just perturb the starting position of the ray, rather than direction.
    """
    def __init__(self, rays, perturbations):
        """rays is a list of Rays; perturbations is an n x 4 x 2 tensor."""
        self.rays = rays
        self.perturbations = perturbations

    # @classmethod
    # def of_transformation_bad(cls, M, epsilon, wavelength=green):
    #     """M is transformation that gives starting position of ray.  It is applied to a RayBundle
    #     that starts at origin and points in the direction (0,0,-1).
    #     """
    #     num_rays = 3
    #     p = np.zeros((num_rays,4,2))
    #     # TODO: More perturbations.  Share code with CircularSource?  It's kind of similar.
    #     p[1,:,1] = np.array([-epsilon, 0., 0., 1.])
    #     p[2,:,1] = np.array([0., -epsilon, 0., 1.])
    #     basic_ray = np.array([[0., 0.],
    #                           [0., 0.],
    #                           [-1., 0.],
    #                           [0., 1.]])
    #     ray_positions = basic_ray + p

    #     # Rather than put all the rays in a single tensor, we'll just have a list of
    #     # vanilla rays because other code expects those.
    #     rays = [Ray(ray_positions[i,:,:], 0., wavelength) for i in range(num_rays)]
    #     untransformed = cls(rays, p)
    #     return untransformed.transform(M)

    # TODO: It's kind of awkward that in some places k is considered default direction for
    # a ray to point while here it is -k.
    @classmethod
    def of_transformation(cls, M, epsilon, wavelength=green):
        """M is transformation that gives starting position of ray.  It is applied to a RayBundle
        that starts at origin and points in the direction (0,0,-1).
        """
        # CircularSource points in direction k by default; we want -k.
        rays, _pairs = CircularSource(np.diag([1,1,-1,1]), epsilon).get_rays(np.eye(4))
        num_rays = len(rays)
        perturbations = np.zeros((num_rays,4,2))
        for i, ray in enumerate(rays):
            perturbations[i] = ray.v_q
        rays = [ ray.transform(M) for ray in self.rays ]
        return cls(rays, perturbations)

    def transform(self, M):
        assert M.shape == (4,4), f"bad M = {M}"

        rays = [ray.transform(M) for ray in self.rays]

        # TODO: It's currently moot, but maybe this should just be RayBundle rather than
        # self.__class__.  Supposing hypothetically that we subclassed this, and called this
        # method, would we want an element of the subclass?  On one hand it seems like we would;
        # on the other hand, we don't know what arguments the subclass's constructor would take,
        # so self.__class__(...) seems unsound without an explicit convention that subclasses
        # must have constructors that take the same arguments.
        return self.__class__(rays, self.perturbations)

    def cointeract(self, element):
        """element should have an interact method that expects a ray, and returns another ray
        or None

        This is kind of dual to interact--it is the element that is interacting with our rays.
        """
        rays = [ element.interact(ray) if ray is not None else None for ray in rays ]
        return self.__class__(rays, self.perturbations)

    def approx_focus(self):
        """We solve for t that would cause resulting points to be as close as possible
        to each other.  Then we take the centroid for that value of t.

        The approach is that, for a given t, there is a sum of squared distances from
        the centroid.  I think this will be a quadratic function of t, which we can
        then minimize.

        (This, in some sense, cares about rays being close in spacetime.  You could
        imagine an alternative approach where each ray traces out a line, and we look
        for the point in space whose sum of squared distances from these lines is as small
        as possible.  This isn't hard either, since given a line, there's a quadratic
        function that gives squared distance from it.)

        Returns the approximate focal point, and dictionary with additional info:
        - t
        - rms (of distance of rays from that point at time t)
        - possibly further information about kinds of aberration (coma, spherical
          aberration, astigmatism); if rays come with a color, and we perturb wavelength,
          then we could also get chromatic aberration.
        """
        num_rays = len(self.rays)
        rays = np.zeros((num_rays,4,2))
        for i, ray in enumerate(self.rays):
            rays[i,:,:] = ray.zero_phase_v_q()
        centroid_ray = rays.mean(axis=0)
        diffs_from_centroid = rays - centroid_ray
        # The following could also be done with einsum.
        #accum = np.einsum('ijk,ijl->kl', diffs_from_centroid, diffs_from_centroid)
        accum = np.zeros((2,2))
        num_rays = diffs_from_centroid.shape[0]
        for i in range(num_rays):
            d = diffs_from_centroid[i,:,:]
            accum += d.T.dot(d)
        # [t;1]' * accum * [t;1] is the loss we want to minimize.
        # a t^2 + b t + c has deriv 2 a t + b and (assuming a > 0)
        # is minimized at t=-b/2a
        a = accum[0,0]
        b_over_2 = accum[0,1]
        if a <= 0:
            t = np.nan
        else:
            t = -b_over_2 / a
        t1 = np.array([t, 1.])
        approx_focus = centroid_ray.dot(t1)
        rms = np.sqrt(np.einsum('i,ij,j', t1, accum, t1))
        return approx_focus, { 't': t, 'rms': rms }

    def approx_focal_length(self):
        """If we assume the perturbations were just spatial, then how those perturbations
        relate to differences in angle of the rays should give us information about the focal
        length of whatever system the RayBundle went through.

        (If perturbations weren't spatial, I think we can't do it.  E.g., suppose they
        were perturbations of angle, and we find that a divergence of dtheta=0.01 became
        a divergence of dtheta=-0.01.  There are no distance units there so no information
        about distance.  But perhaps if we took into consideration positions of rays
        and values of t...)
        """
        # TODO: implement
        assert False, "Not yet implemented"


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
class OrientedSurface:
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

