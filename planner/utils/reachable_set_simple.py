"""Implementation of simple reachable set over-approximation."""

import numpy as np
import math
import shapely.geometry
from EthicalTrajectoryPlanning.planner.utils import shapely_conversions


class ReachSetSimple(object):
    """
    Class for calculation of simple reach set over-approximation.

    Class for calculation of simple reach set over-approximation.
    Provides option to trim the reachable set to provided
    track boundaries (provide via init call).

    If this functionality is not required, calling the
    "simple_reachable_set()"-method is sufficient
    (no class needed).

    """

    def __init__(
        self,
        closed: bool = False,
        bound_l: np.ndarray = None,
        bound_r: np.ndarray = None,
    ) -> None:
        """
        Initilize a simple reachable set.

        :param closed:  flag indicating whether track bounds
                        are a closed circuit or not
        :param bound_l: coordinates of the left bound
                        (numpy array with columns x, y)
        :param bound_r: coordinates of the right bound
                        (numpy array with columns x, y)

        """
        # calculate patch of track
        # (in order to trim to bounds by calculating
        # intersection with reachable set)
        if bound_l is not None and bound_r is not None:
            if closed:
                polygon_l = shapely.geometry.Polygon(bound_l)
                polygon_r = shapely.geometry.Polygon(bound_r)

                if polygon_l.area > polygon_r.area:
                    self.__intersection_patch = polygon_l.difference(polygon_r)
                else:
                    self.__intersection_patch = polygon_r.difference(polygon_l)
            else:
                self.__intersection_patch = shapely.geometry.Polygon(
                    np.row_stack((bound_l, np.flipud(bound_r)))
                )

        else:
            self.__intersection_patch = None

    def calc_reach_set(
        self,
        obj_pos: np.ndarray,
        obj_heading: float,
        obj_vel: float,
        obj_length: float,
        obj_width: float,
        dt: float,
        t_max: float,
        a_max: float,
    ) -> dict:
        """
        Calculate a simple reachable set approximation, trimmed to bounds.

        Calculate a simple reachable set approximation for an object,
        based on its position and speed. '`dt`' and '`t_max`'
        define the resolution and the number of time-steps.

        :param obj_pos:     position of the vehicle (x and y coordinate)
        :param obj_heading: heading of the vehicle
        :param obj_vel:     velocity of the vehicle
        :param obj_length:      length of the vehicle [in m]
        :param obj_width:       width of the vehicle [in m]
        :param dt:          desired temporal resolution of the reachable set
        :param t_max:       maximum temporal horizon for the reachable set
        :param a_max:       maximum assumed acceleration of the object vehicle
        :returns:
            * **poly** -    dict of reachable areas with:

                * keys holding the evaluated time-stamps
                * values holding the outlining coordinates as
                    a np.ndarray with columns [x, y]

        """
        # calculate reachable set
        srs = simple_reachable_set(
            obj_pos=obj_pos,
            obj_heading=obj_heading,
            obj_vel=obj_vel,
            obj_length=obj_length,
            obj_width=obj_width,
            dt=dt,
            t_max=t_max,
            a_max=a_max,
        )

        # if provided, trim reachable set to track
        if self.__intersection_patch is not None:
            for t_key in srs.keys():
                srs_t = shapely.geometry.Polygon(srs[t_key])

                srs_t = srs_t.intersection(self.__intersection_patch)

                # if coordinates present (not wiped out completely),
                # extract outline coordinates
                red_tmp = shapely_conversions.extract_polygon_outline(
                    shapely_geometry=srs_t
                )

                # add outline coordinates to reach set
                if red_tmp is not None:
                    srs[t_key] = red_tmp

        return srs

    @property
    def intersection_patch(self):
        """
        Get intersection patch.

        Returns:
            The intersection patch.
        """
        return self.__intersection_patch


def simple_reachable_set(
    obj_pos: np.ndarray,
    obj_heading: float,
    obj_vel: float,
    obj_length: float,
    obj_width: float,
    dt: float,
    t_max: float,
    a_max: float,
) -> dict:
    """
    Calculate a simple reachable set approximation.

    Calculate a simple reachable set approximation for an object,
    based on its position and speed. '`dt`' and '`t_max`'
    define the resolution and the number of time-steps.

    :param obj_pos:     position of the vehicle (x and y coordinate)
    :param obj_heading: heading of the vehicle
    :param obj_vel:     velocity of the vehicle
    :param obj_length:      length of the vehicle [in m]
    :param obj_width:       width of the vehicle [in m]
    :param dt:          desired temporal resolution of the reachable set
    :param t_max:       maximum temporal horizon for the reachable set
    :param a_max:       maximum assumed acceleration of the object vehicle
    :returns:
        * **poly** -    dict of reachable areas with:

            * keys holding the evaluated time-stamps
            * values holding the outlining coordinates
                as a np.ndarray with columns [x, y]

    :Authors:
        * Yves Huberty
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        27.01.2020

    """
    # executing motion model on vehicle
    cvehicle = cv_model(vel=obj_vel, dt=dt, t_max=t_max)

    # executing motion model on vehicle
    bx_bound = bx_boundary(vel=obj_vel, a_max=a_max, dt=dt, t_max=t_max)

    # calculate radius of reachable area
    racc = calc_acc_rad(amax=a_max, dt=dt, t_max=t_max)

    # global angle
    global_angle = obj_heading  # + math.pi / 2

    # calculate vertices
    poly = calc_vertices(
        pos=obj_pos,
        cvehicle=cvehicle,
        bx_bound=bx_bound,
        globalangle=global_angle,
        racc=racc,
        dt=dt,
        veh_length=obj_length,
        veh_width=obj_width,
    )

    return poly


def polar2cart(radius: float, angle: float) -> np.ndarray:
    """
    Transform polar coordinates to cartesian coordinates.

    Transform polar coordinates to cartesian coordinates
    and returns the values in the vector format.

    :param radius:      radius of polar coordinate
    :param angle:       angle of polar coordinate
    :returns:
        * **coord** -   x, y values of point converted to cartesian coordinates

    """
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)

    return np.array([[x], [y]])


def rotate_vector(vector: np.array, angle: float) -> np.ndarray:
    """
    Rotates a vector by the angle alpha around the origin.

    :param vector:      vector
    :param angle:       angle in rad
    :returns:
        * **vector** -  rotated vector

    """
    s = math.sin(angle)
    c = math.cos(angle)

    # Rotate vector
    xnew = c * vector[0] - s * vector[1]
    ynew = s * vector[0] + c * vector[1]

    return np.array([xnew, ynew])


def cv_model(vel: float, dt: float, t_max: float) -> np.ndarray:
    """
    CV-Model for prediction of the center of the vehicle.

    CV-Model for prediction of the center of the vehicle
    (along vehicle's longitudinal axis).

    :param vel:             velocity of the vehicle [in m/s]
    :param dt:              temporal increment between pose predictions [in s]
    :param t_max:           precition horizon [in s]
    :returns:
        * **cvehicle** -    predicted centers of the vehicle

    """
    # calculate cvehicle
    cvehicle = vel * np.arange(0.0, t_max + dt / 2, dt)
    cvehicle = np.transpose(cvehicle)

    return cvehicle


def bx_boundary(vel: float, a_max: float, dt: float, t_max: float) -> np.ndarray:
    """
    Calculate boundary (b_x).

    Calculate boundary (b_x) according to:
    "Set-Based Prediction of Traffic Participants on Arbitrary Road Networks"
    by M. Althoff and S. Magdici -> Eq. (4)


    :param vel:             velocity of the vehicle [in m/s]
    :param a_max:           maximum acceleration [in m/s²]
    :param dt:              temporal increment between pose predictions [in s]
    :param t_max:           prediction horizon [in s]
    :returns:
        * **bx_bound** -    bx_boundary

    """
    # over-approximate velocity if close to zero
    if math.isclose(vel, 0.0, rel_tol=0.0, abs_tol=0.001):
        vel = 0.01

    # calculate maximum value
    t_bmax = np.sqrt(2 / 3) * vel / a_max
    bx_bound_max = vel * t_bmax - a_max * a_max * np.power(t_bmax, 3) / (2 * vel)

    # calculate bx_bound
    t = np.arange(0.0, t_max + dt / 2, dt)
    bx_bound = vel * t - a_max * a_max * np.power(t, 3) / (2 * vel)
    bx_bound = np.transpose(bx_bound)

    # keep bx_bound_max, when max. time is exceeded
    bx_bound[t > t_bmax] = bx_bound_max

    return bx_bound


def calc_acc_rad(amax: float, dt: float, t_max: float) -> list:
    """
    Calculate the radius of the largest reachable set.

    :param amax:        maximum acceleration [in m/s^2]
    :param dt:          prediction resolution [in s]
    :param t_max:       precition horizon [in s]
    :returns:
        * **racc** -    sequence of radii
    """
    racc = []
    for t in np.arange(0.0, t_max + dt / 2, dt):
        racc.append(0.5 * amax * t ** 2)
    return racc


def calc_vertices(
    pos: np.ndarray,
    cvehicle: np.ndarray,
    bx_bound: np.ndarray,
    globalangle: float,
    racc: list,
    dt: float,
    veh_length: float = 0.0,
    veh_width: float = 0.0,
) -> dict:
    """
    Calculate the vertices of the enveloping polygon.

    .. code-block::

              q2-----q3
            -        |
        q1           |
         |           |
        q6           |
            -        |
              q5-----q4

    Calculation of over-approximation polygon for each step.
    Method based on "SPOT: A Tool for Set-Based Prediction of
    Traffic Participants" by M. Koschi and M. Althoff


    :param pos:              position of the vehicle [in m]
    :param cvehicle:         center positions of vehicle
                             at future time-stamps (along longitudinal axis)
    :param bx_bound:         front extension of overapprox.
                             reach set (based on Althoff)
    :param globalangle:      angle [in rad]
    :param racc:             radius of reachable area
                             at certain points in time [in m]
    :param dt:               temporal increment between pose predictions [in s]
    :param veh_length:       (optional) vehicle length [in m],
                             if not provided, reach-set for
                             point-mass is calculated
    :param veh_width:        (optional) vehicle width [in m],
                             if not provided, reach-set for
                             point-mass is calculated
    :returns:
        * **polypred** -     dict of reachable areas with:

            * keys holding the evaluated time-stamps
            * values holding the outlining coordinates
                as a np.ndarray with columns [x, y]

    :Authors:
        * Yves Huberty
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        27.01.2020

    """
    # initialize empty polygon dict
    polypred = {}

    # init vehicle dimension array
    veh_dim = (
        np.array(
            [
                [-veh_length, veh_width],
                [-veh_length, veh_width],
                [veh_length, veh_width],
                [veh_length, -veh_width],
                [-veh_length, -veh_width],
                [-veh_length, -veh_width],
            ]
        )
        / 2.0
    )

    prev_front = -99.0
    prev_r_t = 0.0
    for j in range(cvehicle.shape[0]):
        # retrieve relevant parameters
        if j > 0:
            r_t = racc[j - 1]
            c_t = cvehicle[j - 1]
            b_t = bx_bound[j - 1]
        else:
            r_t = 0.0
            c_t = 0.0
            b_t = 0.0
        r_t1 = racc[j]
        c_t1 = cvehicle[j]

        # calculate basic polygon (in plane)
        poly = np.array(
            [
                [c_t - r_t, r_t],  # q1
                [b_t, r_t1],  # q2
                [c_t1 + r_t1, r_t1],  # q3
                [c_t1 + r_t1, -r_t1],  # q4
                [b_t, -r_t1],  # q5
                [c_t - r_t, -r_t],
            ]
        )  # q6

        # prevent from driving backwards when reaching v=0
        if poly[0, 0] < prev_front:
            poly[0, 0] = prev_front
            poly[0, 1] = prev_r_t
            poly[-1, 0] = prev_front
            poly[-1, 1] = -prev_r_t

        else:
            prev_front = poly[0, 0]
            prev_r_t = poly[0, 1]

        # add vehicle dimensions
        poly = poly + veh_dim

        # rotate and translate
        for i in range(np.shape(poly)[0]):
            poly[i, :] = rotate_vector(poly[i, :], globalangle) + np.transpose(pos)

        # add to polypred
        polypred[round(dt * j, 2)] = poly

    return polypred
