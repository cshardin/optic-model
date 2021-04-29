from __future__ import print_function, division # in case used from python2

import numpy as np
import numpy.linalg as LA

from common import *
from geometry import Ray, Quadric, Plane, translation3f, point, vector, cross, ii, jj, kk, make_rotation_y, make_bound_vector, CircularSource, LinearSource
from elements import SubElement, Compound, make_conic, make_lens_basic, make_lens

class CircularAperture:
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

class PlanarSensor:
    """
    A flat sensor (like a camera sensor).  Doesn't have finite extent.

    You probably want to call PlanarSensor.of_q_x_y to construct.
    """
    # TODO: Reverse the convention for x_dir or y_dir.  We want it to be such that if you "took
    # a picture" of a set of x and y axes using the scope, then they'd look like x and y axes
    # in the picture.  So, for a pinhole camera, x_dir should be (-1, 0, 0) while y_dir should
    # be (0, -1, 0).  (Note: It's easy to get this backwards in your head.  Perhaps think of it
    # this way: if you are standing out on the positive z-axis, looking in the direction of
    # the negative z-axis--i.e., you are looking at the telescope and the telescope is looking
    # at you--the x- and y-axes look to you like they would as you normally draw
    # them on paper.  The image you seen projected onto the sensor will be upside down and
    # backwards.  Note that if you actually took a picture with this instrument, and drew
    # with x- and y- as you ordinarily draw these axes, the picture will be a mirror image.)
    # We also want to be able to infer which way the sensor is facing from x_dir and y_dir;
    # for a design with a single reflector (and no shenanigans like the mirror being trough-shaped
    # so that it "inverts" on one axis but not the other), this means flipping sign of the direction
    # it's facing.  What's really going on is that in such cases, the M-matrix we are forming
    # should have determinant -1.  I think we should add a boolean `flip` argument that says
    # that x_dir x y_dir needs to be flipped to get the direction the sensor is facing.
    # (In principle we really want the user to specify an orthogonal set of axes.  But it's
    # simpler for caller to just take in x and y and a sign for z = +/- x cross y.)

    def __init__(self, M):
        """
        M is that matrix that moves xyz axis to where sensor is.  In the resulting object,
        M_inv is the matrix that transforms a ray into the sensor's coordinate system.
        """
        debug = True
        M_inv = LA.inv(M) # should be numerically stable
        self.M = M # in case we need to, say, draw a little picture of the sensor
        self.M_inv = M_inv # the thing we actually work with in most calculations
        if debug:
            print(f"made sensor M={self.M}, M_inv={self.M_inv}")

    @staticmethod
    def of_q_x_y(q, x_dir, y_dir, flip_z=False):
        """
        x_dir and y_dir should be orthogonal unit vectors unless you're doing something
        unusual (in which case you're on your own).  z_dir = x_dir x y_dir by default,
        but this is negated if flip_z=True.  z_dir is the direction the sensor is facing.
        """
        sign = -1 if flip_z else 1
        z_dir = sign * cross(x_dir, y_dir)
        M = np.stack([x_dir, y_dir, z_dir, q], axis=1)
        return PlanarSensor(M)

    def setback(self, offset):
        """Positive offset means move sensor back (away from the direction its pointing)"""
        M = self.M.dot(translation3f(0, 0, -offset))
        return PlanarSensor(M)

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
            if debug:
                print(f"caught at point {x,y} with phase {phase}")
            return (x,y,phase)

class Instrument:
    """An optical device: a sequence of reflectors and lenses or other elements"""
    # TODO: change name of elements to a singular, like element_group
    def __init__(self, source, elements, sensor):
        self.source = source
        self.elements = elements
        self.sensor = sensor

    def setback_sensor(self, setback):
        """A positive setback means move sensor away from the direction it is pointed."""
        return Instrument(self.source, self.elements, self.sensor.setback(setback))

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

# TODO: Move to elements?
def make_paraboloid(focal_length, material=None):
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
    geometry = Quadric(M)
    return SubElement(geometry, material=material)


def standard_source(z, radius):
    """A source pointing "down" (direction of rays is (0,0,-1)) from (0,0,z) with given radius"""
    debug = False
    if debug:
       return LinearSource(radius, z)
    source_orientation = np.diag([-1,1,-1,1])
    source_T = translation3f(0,0,z).dot(source_orientation)
    source = CircularSource(source_T, radius)
    return source

def make_newtonian(focal_length, aperture_radius):
    """A very simple Newtonian design where we don't even stick in the flat secondary, because
    we're just trying to analyze coma (which isn't affected by the flat secondary).
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
    sensor = PlanarSensor.of_q_x_y(point(0,0,focal_length), -ii, -jj, flip_z=True)
    elements = Compound([aperture0, aperture1, reflector])
    return Instrument(source, elements, sensor)

def make_simple_refractor(focal_length, aperture_radius, d):
    """Refractor with single lens

    d is thickness of lens
    """
    # TODO: Check whether the value of d gives us a lens consistent with the aperture.
    source = standard_source(d, aperture_radius)
    # We'll think of the aperture as just slightly in front of the lens.
    aperture = CircularAperture(point(0,0,0.75*d), vector(0,0,-aperture_radius))
    sensor = PlanarSensor.of_q_x_y(point(0, 0, -focal_length), -ii, -jj)
    z_offset = 0.
    lens = make_lens_basic(focal_length, d, z_offset)
    elements = Compound([aperture, lens])
    return Instrument(source, elements, sensor)

def make_classical_cassegrain(focal_length, d, b, aperture_radius):
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
    translated_secondary_geometry = Quadric(translated_secondary_M)
    secondary_geometry = translated_secondary_geometry.untransform(translation3f(0,0,-s))
    secondary = SubElement(secondary_geometry, clip=None)
    primary = make_paraboloid(f1)
    aperture0 = CircularAperture(point(0,0,d), vector(0,0,-aperture_radius))
    # Reflector is z = r^2 / (4 focal length), so at edge,
    # we have:
    z_reflector_edge = aperture_radius ** 2 / (4 * f1)
    aperture1 = CircularAperture(point(0,0,z_reflector_edge), vector(0,0,-aperture_radius))

    #source = standard_source(focal_length, aperture_radius)
    source = standard_source(d, aperture_radius)
    elements = Compound([aperture0, aperture1, primary, secondary])
    sensor = PlanarSensor.of_q_x_y(point(0,0,-b), vector(1,0,0), vector(0,1,0))
    return Instrument(source, elements, sensor)

def make_ritchey_chretien(focal_length, D, b, aperture_radius):
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
    primary = make_conic(-R1, K1, 0)
    print(f"primary: {primary}")
    print(f"secondary: trying to make hyperboloid with radius {-R2} conic constant {K2} at z={D}")
    secondary = make_conic(-R2, K2, D, reverse_normal=True)
    print(f"secondary: {secondary}")
    aperture0 = CircularAperture(point(0,0,D), vector(0,0,-aperture_radius))
    # The rule we used for the parabolic should still be approximately correct here.
    z_reflector_edge = aperture_radius ** 2 / (4 * f1)
    aperture1 = CircularAperture(point(0,0,z_reflector_edge), vector(0,0,-aperture_radius))

    #source = standard_source(focal_length, aperture_radius)
    source = standard_source(D, aperture_radius)
    elements = Compound([aperture0, aperture1, primary, secondary])
    sensor = PlanarSensor.of_q_x_y(point(0,0,-b), -ii, -jj)
    return Instrument(source, elements, sensor)

def make_gregory_maksutov(aperture_radius, R1, R2, R3, d, ell, b, lens_material=None):
    """Gregory-Maksutov

    The sign conventions are such that, typically, all parameters below are positive.
    In particular, for the radii of curvature, positive means concave towards positive
    z-axis.

    R1 - radius of curvature of front of lens
    R2 - radius of curvature of back of lens (which is also secondary reflector)
    R3 - radius of curvature of primary reflector
    d - thickness of lens on axis
    ell - distance between primary and secondary reflector
    b - backfocus from primary to sensor
    lens_material - self explanatory; BK7 by default

    """
    if aperture_radius > max(R1, R2, R3):
        raise ValueError("aperture radius must not exceed any radius of curvature")
    # See https://www.cfht.hawaii.edu/~baril/Maksutov/Maksutov.html
    # We'll have primary reflector intersect z-axis at z=0.  I'm not sure if that's
    # the convention we want.
    lens_z_offset = d/2 + ell
    lens = make_lens(-R1, -R2, d, lens_z_offset, material=lens_material)
    # make primary and secondary...
    primary = make_conic(R3, 0., 0.)
    secondary = make_conic(R2, 0., ell)

    # How much farther forward is the rim of the lens from where its front meets
    # z-axis?
    front_depth = R1 - np.sqrt(R1 ** 2 - aperture_radius ** 2)
    # TODO: also include an aperture at primary reflector?
    front_z_coordinate = ell + d + front_depth
    aperture0 = CircularAperture(point(0,0,front_z_coordinate), vector(0,0,-aperture_radius))

    source = standard_source(front_z_coordinate, aperture_radius)
    elements = Compound([aperture0, lens, primary, secondary])
    sensor = PlanarSensor.of_q_x_y(point(0,0,-b), -ii, -jj)
    return Instrument(source, elements, sensor)

def newtonian_example():
    focal_length = 1000 * mm
    aperture_radius = 4 * inch # an 8" scope
    return make_newtonian(focal_length, aperture_radius), focal_length

def classical_cassegrain_example():
    aperture_radius = 4 * inch # an 8" scope
    focal_length = 2000 * mm
    d = 40 * cm
    b = d/2
    return make_classical_cassegrain(focal_length, d, b, aperture_radius), focal_length

def ritchey_chretien_example():
    aperture_radius = 4 * inch # an 8" scope
    focal_length = 2000 * mm
    d = 40 * cm
    b = d/2
    return make_ritchey_chretien(focal_length, d, b, aperture_radius), focal_length

def maksutov_example():
    # Numbers here are mostly just made up.
    # See https://www.telescope-optics.net/maksutov_corrector_radii.htm
    # for choice of R1 = 0.94 * R2.
    aperture_radius = 4 * inch
    R2 = 36 * inch
    R1 = 0.94 * R2
    R3 = 30 * inch
    d = 0.25 * inch
    ell = 12 * inch
    b = 1 * inch
    focal_length = 1000 * mm # not really, just making this up
    return make_gregory_maksutov(aperture_radius, R1, R2, R3, d, ell, b), focal_length

def simple_refractor_example():
    aperture_radius = 2 * inch # a 4" scope
    focal_length = 1000 * mm
    d = 1 * cm
    return make_simple_refractor(focal_length, aperture_radius, d), focal_length
