#!/usr/bin/env python3
"""Analyze optical systems

This is vaguely like a ray tracer, but:
- Elements are encountered in a prescribed order.
- We track phase information (and maybe more stuff later).
- We trace from origin to a detector.

Units are generally meters, though currently it doesn't matter--they're just abstract distance units.

Instruments generally point in direction of positive z-axis.

We use homogeneous coordinates (x,y,z,w) and the usual way of using 4x4 matrices for transformations
(as in OpenGL, for instance).  There might be places where we do some tricks that interpret this as
the quaternion w + x i + y j + z k.
"""
# TODO: Add lenses and ability to track what happens to different wavelengths.
# TODO: Handle polarization information.  Is this just going to
# be some matrix?
from __future__ import print_function, division # in case used from python2...

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

inch = 0.0254
mm = 1e-3
cm = 1e-2
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

class Quadric():
    def __init__(self, M, ior=None):
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
        self.ior = ior

    def __str__(self):
        return f"{self.M} (ior={self.ior})"

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

    def reflect(self, ray):
        t = self.intersect(ray)
        if t is None:
            return None
        R = ray.v_q
        new_q = R[:,0] * t + R[:,1]
        phase = ray.phase + t
        grad = self.M.dot(new_q)
        # TODO: Is this right to just zero out last coordinate of grad to
        # make it vector-like?  It feels a little unnatural...
        grad[3] = 0
        v = R[:,0]
        v_dot_grad = v.dot(grad)
        assert v_dot_grad < 0, "thought gradient pointed towards us"
        # To reflect, we subtract twice the projection onto grad.
        p_v_onto_grad = (v_dot_grad / grad.dot(grad)) * grad
        reflected_v = v - 2 * p_v_onto_grad
        assert reflected_v[3] == 0., f"bad reflection {reflected_v} (not vector-like); v={v}; grad={grad}"
        new_v_q = np.stack([reflected_v, new_q], axis=1)
        return Ray(new_v_q, phase, ray.annotations)

    def interact(self, ray):
        """Compute reflected/transmitted ray"""
        if self.ior is None:
            return self.reflect(ray)
        else:
            raise NotImplementedError()

    def untransform(self, A):
        """Apply inverse of A to the quadric."""
        return Quadric(A.T.dot(self.M).dot(A), self.ior)

class CircularAperture():
    def __init__(self, q, v):
        """q is location, v is normal, with length giving radius"""
        self.q = q
        self.v = v

    def interact(self, ray):
        """To interact, we either absorb the ray or don't.  We won't "move" the photon
        up to the aperture"""
        # *Should* we move it up to the aperture?  If you design contrived stuff, like the aperture being
        # behind a reflector, then our approach gives non-physical results.
        # But for how we're using it, we like behavior like this--in particular, we like it to be able to
        # block a photon that is already past it (b/c the way we're using it, that's just because we
        # didn't bother to make the photon start farther back).
        # TODO: implement properly; for now we just never absorb
        return ray

class Compound():
    """Just a bunch of elements sequentially"""
    def __init__(self, elements):
        self.elements = elements

    def interact(self, ray):
        debug = True
        if debug:
            print(f"interact got ray {ray.v_q}")
        for index, elt in enumerate(self.elements):
            ray = elt.interact(ray)
            if debug:
                if ray is not None:
                    print(f"{index}: {ray.v_q}")
                else:
                    print(f"{index}: {ray}")
            if ray is None:
                return None
        return ray

class CircularSource():
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
        pairs.append((0,1))
        # Let's say we have 1 point in center, 6 points around that, then 12 points
        # around that; we want the 4th point of those 12; so it should have 1 + 6 + (12 / 4)
        # points before it.
        pairs.append((0, 1 + resolution + (resolution // 2)))
        if debug:
            print(f"pairs={pairs}")
            print(f"first few ray points = {[ ray.q() for ray in rays[:3]]}")
        return rays, pairs

class LinearSource():
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

class PlanarSensor():
    def __init__(self, q, x_dir, y_dir):
        """A flat sensor (like a camera sensor).  Doesn't have finite extent.
        x_dir and y_dir should be unit vectors.
        """
        debug = True
        z_dir = cross(x_dir, y_dir)
        M = np.stack([x_dir, y_dir, z_dir, q], axis=1)
        M_inv = LA.inv(M) # should be numerically stable
        self.M = M # in case we need to, say, draw a little picture of the sensor
        self.M_inv = M_inv # the thing we actually work with in most calculations
        if debug:
            print(f"made sensor M={self.M}, M_inv={self.M_inv}")

    def catch(self, ray):
        """Return point in sensor's coordinate system where ray hits, and the ray's phase.
        Return None if the ray misses the sensor.

        TODO: For some calculations, we'll probably also care about the angle it hits at.
        """
        debug = True
        if debug:
            speed = LA.norm(ray.v_q[:,0])
            print(f"catching ray {ray.v_q} with speed {speed}")
        ray = ray.transform(self.M_inv)
        if debug:
            print(f"transformed to {ray.v_q}")
        # Now the ray is in the sensor's local coordinates.
        ray_q = ray.q()
        ray_z = ray_q[2] #ray's z coordinate
        ray_v = ray.v()
        ray_zdot = ray_v[2] #ray's z velocity
        if ray_zdot == 0:
            if debug:
                print("ray_zdot == 0")
            return None # parallel to sensor counts as miss even if "in" sensor
        t = -ray_z / ray_zdot
        if t < 0:
            if debug:
                print(f"t = {t}, ray_z = {ray_z}, ray_zdot = {ray_zdot}")
            return None
        else:
            ray = ray.advance_time(t)
            ray_q = ray.q()
            x = ray_q[0]
            y = ray_q[1]
            phase = ray.phase
            # Should we return a 4-vector rather than x,y?
            return (x,y,phase)

class Instrument():
    """An optical device: a sequence of reflectors and lenses or other elements"""
    # TODO: change name of elements to a singular, like element_group
    def __init__(self, source, elements, sensor):
        self.source = source
        self.elements = elements
        self.sensor = sensor

    def simulate(self, R):
        """Simulate rays from the source, if it is rotated according to R."""
        debug = True
        rays, pairs = self.source.get_rays(R)
        caught = []
        for ray in rays:
            # print(f"type(ray)={type(ray)}")
            ray = self.elements.interact(ray)
            result = self.sensor.catch(ray)
            if result is not None:
                (x,y,phase) = result
                if debug:
                    print(f"ray {ray.v_q} hit detector at {x,y}")
                caught.append((x,y))
            else:
                if debug:
                    print(f"ray {ray.v_q} missed detector")
                # TODO: bug: if anything misses detector, indices of pairs are wacky
                #caught.append(None)
        return caught, pairs

def make_paraboloid(focal_length, ior=None):
    """
    A parabolic reflect with back of the reflector at (0,0,0), with normal (0,0,1).
    Focal point is (0,0,focal_length).
    The paraboloid then looks like z = alpha r^2 where alpha = 1 / (4 focal length).
    dz/dr = 2 alpha r = 1 when r = 1/(2 alpha), at which point z = alpha / (4 alpha^2) = 1 / (4 alpha).
    So, we want focal_length = 1 / (4 alpha), so alpha = 1 / (4 focal length).
    It turns out it will be easier to think of this as 4 fl z = r^2 (where fl = focal_length)

    So as a quadratic form we have 4 fl z = x^2 + y^2
    x^2 + y^2 - 4 fl z = 0
    So the matrix is
    [1 0 0 0
     0 1 0 0
     0 0 0 -2*fl
     0 0 -2*fl 0]
    but we negate this because we want normal vector at (0,0,0) pointing in direction (0,0,1), not (0,0,-1).
    """
    alpha = 1 / (4 * focal_length)
    M = np.zeros((4,4))
    M[0,0] = -1
    M[1,1] = -1
    M[2,3] = 2 * focal_length
    M[3,2] = M[2,3]
    return Quadric(M, ior=ior)

# TODO: Should make_paraboloid and make_hyperboloid be class functions of Quadric?
# TODO: Does the following just do the right thing for other conics, not just
# hyperboloids?  If so, change name to make_conic?
# TODO: Our convention for using normals to determine which surface you hit is kind of annoying...
def make_hyperboloid(R, K, z_offset, ior=None, reverse_normal=False):
    """
    See https://en.wikipedia.org/wiki/Conic_constant

    r^2 - 2Rz + (K+1)z^2 = 0

    Args:
        R: radius of curvature; use R > 0 for concave "up" while R < 0 is concave "down"
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
    quad = Quadric(M, ior=ior)
    return quad.untransform(translation3f(0,0,-z_offset))

def standard_source(z, radius):
    """A source pointing "down" from (0,0,z) with given radius"""
    debug = False
    if debug:
       return LinearSource(radius, z)
    source_orientation = np.diag([-1,1,-1,1])
    source_T = translation3f(0,0,z).dot(source_orientation)
    source = CircularSource(source_T, radius)
    return source

def make_newtonian(focal_length, aperture_radius, setback=0.):
    """A very simple Newtonian design where we don't even stick in the flat secondary, because
    we're just trying to analyze coma (which isn't affected by the flat secondary).

    setback is how far we push the sensor from where it "should" be.
    """
    source = standard_source(focal_length, aperture_radius)
    aperture0 = CircularAperture(point(0,0,focal_length), vector(0,0,-aperture_radius))
    # Reflector is z = r^2 / (4 focal length), so at edge,
    # we have:
    z_reflector_edge = aperture_radius ** 2 / (4 * focal_length)
    aperture1 = CircularAperture(point(0,0,z_reflector_edge), vector(0,0,-aperture_radius))
    reflector = make_paraboloid(focal_length)
    # In practice there would be another mirror.  So we actually pretend the sensor
    # is facing up.
    sensor = PlanarSensor(point(0,0,focal_length + setback), vector(1,0,0), vector(0,1,0))
    elements = Compound([aperture0, aperture1, reflector])
    return Instrument(source, elements, sensor)

def make_classical_cassegrain(focal_length, d, b, aperture_radius, setback=0.):
    """A classical Cassegrain scope: parabolic primary, hyperbolic secondary.

    One thing that's a little tricky is it's hard to infer the parameters from
    product descriptions (not that they're classical Cassegrains, but still...).
    Eyeballing some pictures it looks like b is typically about half of d.
    Note that d is significantly shorter than total length of the tube.

    Args:
        focal_length: focal length
        d: distance between primary and secondary (along main axis)
        b: backfocus (how far behind the primary the focal plane is)
        aperture_radius: aperture radius
        #f1: focal length of primary reflector
        setback: how far to move sensor from where it "should" be

    Returns:
        The resulting Instrument

    """
    # Our notation follows https://ccdsh.konkoly.hu/wiki/Cassegrain_Optics
    # Some math:
    # M := secondary magnification = f / f1 where f1 is focal length of primary
    # M should also be the ratio of the distances to the two focal points from
    # the secondary.
    # Our actual focal plane is at z = -b (with back of primary at z=0).
    # The secondary is at z = d.
    # So consider a proposed f1; then one focal point is at z=f1.
    # The two distances to the foci are d+b and (f1-d), so
    # M = (d+b)/(f1-d)
    # so f1 = d + (d + b)/M, one of the formulas we see on that page.
    # This yields f1 = d + (d+b)/(f/f1) = d + f1(d+b)/f
    # (1 - (d+b)/f) f1 = d
    # f1 = d / (1 - (d+b)/f)
    # Thinking in the (r,z) plane, we want the hyperbola through the point
    # (0,d) with foci (0,f1) and (0,-b)
    # The hyperbola is centered at z=(f1 - b)/2 =: s (our notation)
    # Let z' = z - s; switching to the (r,z') plane, the hyperbola goes through
    # (0, d-s) with foci (0, f1 - s) and (0, -(f1-s)).
    # Let c = f1 - s, a = d - s
    # We'll have z'^2 / a^2 - r^2/beta^2 = 1
    # Write this as r^2/beta^2 - z'^2/a^2 + 1 = 0 (so gradient points in direction
    # we want).
    # (I'm using beta rather than the b you'll see in a textbook because b is
    # already in scope.)
    # The relationship is c^2 = a^2 + beta^2
    # So, beta^2 = c^2 - a^2
    f = focal_length
    f1 = d / (1 - (d+b)/f)
    # M = f / f1
    # f2 = -(d + b) / (M - 1) # By convention, we'll say the hyperboloid has negative focal length.
    s = (f1 - b)/2
    c = f1 - s
    c_alt = s + b
    print(f"d={d}, b={b}, s={s}, f1={f1}, c={c}, c_alt={c_alt}")
    assert np.isclose(c, c_alt)
    a = d - s
    beta2 = c ** 2 - a ** 2

    translated_secondary_M = np.diag([1/beta2, 1/beta2, -1/(a**2), 1])
    translated_secondary = Quadric(translated_secondary_M, ior=None)
    secondary = translated_secondary.untransform(translation3f(0,0,-s))
    primary = make_paraboloid(f1)
    aperture0 = CircularAperture(point(0,0,d), vector(0,0,-aperture_radius))
    # Reflector is z = r^2 / (4 focal length), so at edge,
    # we have:
    z_reflector_edge = aperture_radius ** 2 / (4 * f1)
    aperture1 = CircularAperture(point(0,0,z_reflector_edge), vector(0,0,-aperture_radius))

    #source = standard_source(focal_length, aperture_radius)
    source = standard_source(d, aperture_radius)
    elements = Compound([aperture0, aperture1, primary, secondary])
    sensor = PlanarSensor(point(0,0,-b - setback), vector(1,0,0), vector(0,1,0))
    return Instrument(source, elements, sensor)

# TODO: I don't like setback being an argument to these functions.  We should
# just create the nominal scope, and then have a method of Instrument that
# lets us move the sensor.  One mild awkwardness is we have a hack to avoid
# introducing the secondary reflector in the Newtonian which leads to an
# ad hoc interpretation of setback.
# TODO: I think this is buggy; I'm getting really screwy results where
# rays are missing the detector.
def make_ritchey_chretien(focal_length, D, b, aperture_radius, setback=0.):
    """Ritchey-Chrétien

    Args:
        focal_length: focal length
        D: distance between primary and secondary (along axis)
        b: backfocus from primary to focal plane
        aperture_radius: aperture radius

    Returns:
        The resulting Instrument

    Notation follows https://en.wikipedia.org/wiki/Ritchey%E2%80%93Chr%C3%A9tien_telescope
    """
    debug = True
    F = focal_length
    B = D + b
    R1 = - (2 * D * F) / (F - B)
    R2 = - (2 * D * B) / (F - B - D)
    f1 = np.abs(R1)/2
    f2 = np.abs(R2)/2
    M = F / f1
    M_alt = (F - B) / D
    print(f"M={M}, M_alt={M_alt}")
    assert np.isclose(M, M_alt)

    K1 = -1 - (2 / M ** 3) * (B / D)
    K2 = -1 - (2 / (M - 1) ** 3) * (M * (2 * M - 1) + B/D)

    print(f"primary: trying to make hyperboloid with radius {-R1} conic constant {K1} at z=0")
    primary = make_hyperboloid(-R1, K1, 0)
    print(f"primary: {primary}")
    print(f"secondary: trying to make hyperboloid with radius {-R2} conic constant {K2} at z={D}")
    secondary = make_hyperboloid(-R2, K2, D, reverse_normal=True)
    print(f"secondary: {secondary}")
    aperture0 = CircularAperture(point(0,0,D), vector(0,0,-aperture_radius))
    # The rule we used for the parabolic should still be approximately correct here.
    z_reflector_edge = aperture_radius ** 2 / (4 * f1)
    aperture1 = CircularAperture(point(0,0,z_reflector_edge), vector(0,0,-aperture_radius))

    #source = standard_source(focal_length, aperture_radius)
    source = standard_source(D, aperture_radius)
    elements = Compound([aperture0, aperture1, primary, secondary])
    sensor = PlanarSensor(point(0,0,-b - setback), vector(1,0,0), vector(0,1,0))
    return Instrument(source, elements, sensor)

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

def newtonian_example(setback):
    focal_length = 1000 * mm
    aperture_radius = 4 * inch # an 8" scope
    return make_newtonian(focal_length, aperture_radius, setback=setback), focal_length

def classical_cassegrain_example(setback):
    aperture_radius = 4 * inch # an 8" scope
    focal_length = 2000 * mm
    d = 40 * cm
    b = d/2
    return make_classical_cassegrain(focal_length, d, b, aperture_radius, setback=setback), focal_length

def ritchey_chretien_example(setback):
    aperture_radius = 4 * inch # an 8" scope
    focal_length = 2000 * mm
    d = 40 * cm
    b = d/2
    return make_ritchey_chretien(focal_length, d, b, aperture_radius, setback=setback), focal_length

# Nots: On classical cassegrain, caught1 has +/- 1e-5 in y direction, similar in x direction but coma-like
def test0():
    #setback = 5 * mm # to see how it affects focus
    setback = 0 * mm
    #instrument, focal_length = newtonian_example(setback)
    #instrument, focal_length = classical_cassegrain_example(setback)
    instrument, focal_length = ritchey_chretien_example(setback)

    R0 = np.eye(4) # head on
    caught0, pairs0 = instrument.simulate(R0)

    # Now, rotate slightly around y-axis.
    # Note that a 35mm sensor is 24mm x 36mm.  So, let us consider an angle that puts us roughly
    # 8mm off the center of the sensor and then 16mm off the center.
    theta = 8 * mm / focal_length
    R1 = make_rotation_y(theta)
    caught1, pairs1 = instrument.simulate(R1)

    theta *= 2
    R2 = make_rotation_y(theta)
    caught2, pairs2 = instrument.simulate(R2)

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

def test1():
    #setback = 5 * mm # to see how it affects focus
    setback = 0 * mm
    #instrument, focal_length = newtonian_example(setback)
    #instrument, focal_length = classical_cassegrain_example(setback)
    instrument, focal_length = ritchey_chretien_example(setback)

    R0 = np.eye(4) # head on
    caught0, pairs0 = instrument.simulate(R0)
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
