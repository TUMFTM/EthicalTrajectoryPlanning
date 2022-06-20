"""This file contains relevant functions for the Frenet planner."""

# Standard imports
import os
import sys
import math

# Third party imports
import numpy as np
from scipy.stats import beta
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario

# Custom imports
module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)

from planner.Frenet.utils.polynomials import quartic_polynomial, quintic_polynomial
from planner.Frenet.utils.validity_checks import VALIDITY_LEVELS, check_validity
from planner.Frenet.utils.calc_trajectory_cost import (
    calc_trajectory_costs,
    distance,
)
from planner.utils.timers import ExecTimer
from planner.Frenet.utils.helper_functions import get_max_curvature
from EthicalTrajectoryPlanning.risk_assessment.risk_costs import calc_risk


class FrenetTrajectory:
    """Trajectory in frenet Coordinates with longitudinal and lateral position and up to 3rd derivative. It also includes the global pose and curvature."""

    def __init__(
        self,
        t: [float] = None,
        d: [float] = None,
        d_d: [float] = None,
        d_dd: [float] = None,
        d_ddd: [float] = None,
        s: [float] = None,
        s_d: [float] = None,
        s_dd: [float] = None,
        s_ddd: [float] = None,
        x: [float] = None,
        y: [float] = None,
        yaw: [float] = None,
        v: [float] = None,
        curv: [float] = None,
    ):
        """
        Initialize a frenét trajectory.

        Args:
            t ([float]): List for the time. Defaults to None.
            d ([float]): List for the lateral offset. Defaults to None.
            d_d: ([float]): List for the lateral velocity. Defaults to None.
            d_dd ([float]): List for the lateral acceleration. Defaults to None.
            d_ddd ([float]): List for the lateral jerk. Defaults to None.
            s ([float]): List for the covered arc length of the spline. Defaults to None.
            s_d ([float]): List for the longitudinal velocity. Defaults to None.
            s_dd ([float]): List for the longitudinal acceleration. Defaults to None.
            s_ddd ([float]): List for the longitudinal jerk. Defaults to None.
            x ([float]): List for the x-position. Defaults to None.
            y ([float]): List for the y-position. Defaults to None.
            yaw ([float]): List for the yaw angle. Defaults to None.
            v([float]): List for the velocity. Defaults to None.
            curv ([float]): List for the curvature. Defaults to None.
        """
        # time vector
        self.t = t

        # frenet coordinates
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.d_ddd = d_ddd
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.s_ddd = s_ddd

        # Global coordinates
        self.x = x
        self.y = y
        self.yaw = yaw
        # Velocity
        self.v = v
        # Curvature
        self.curv = curv

        # Validity
        self.valid_level = 0
        self.reason_invalid = None

        # Cost
        self.cost = 0

        # Risk
        self.ego_risk_dict = []
        self.obst_risk_dict = []


def check_curvature_of_global_path(
    global_path: np.ndarray, planning_problem, vehicle_params, ego_state
):
    """
    Check the curvature of the global path.

    If the curvature is to high, points of the global path are removed to smooth the global path. In addition, a new point is added which ensures the initial orientation.

    Args:
        global_path (np.ndarray): Coordinates of the global path.

    Returns:
        np.ndarray: Coordinates of the new, smooth global path.

    """
    global_path_curvature_ok = False

    # get start velocity of the planning problem
    start_velocity = planning_problem.initial_state.velocity

    # calc max curvature for the initial velocity
    max_initial_curvature, _ = get_max_curvature(
        vehicle_params=vehicle_params, v=start_velocity
    )

    # get x and y from the global path
    global_path_x = global_path[:, 0].tolist()
    global_path_y = global_path[:, 1].tolist()

    # add a point to the global path to ensure the initial orientation of the planning problem
    # never delete this point or the initial point
    new_x = ego_state.position[0] + np.cos(ego_state.orientation) * 0.1
    new_y = ego_state.position[1] + np.sin(ego_state.orientation) * 0.1
    global_path_x.insert(1, new_x)
    global_path_y.insert(1, new_y)

    # check if the curvature of the global path is ok
    while global_path_curvature_ok is False:
        # calc the already covered arc length for the points of global path
        global_path_s = [0.0]

        for i in range(len(global_path_x) - 1):
            p_start = np.array([global_path_x[i], global_path_y[i]])
            p_end = np.array([global_path_x[i + 1], global_path_y[i + 1]])
            global_path_s.append(distance(p_start, p_end) + global_path_s[-1])

        # calculate the curvature of the global path
        dx = np.gradient(global_path_x, global_path_s)
        dy = np.gradient(global_path_y, global_path_s)

        ddx = np.gradient(dx, global_path_s)
        ddy = np.gradient(dy, global_path_s)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

        # loop through every curvature of the global path
        global_path_curvature_ok = True
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            if (curvature[i] * 2.0) > max_initial_curvature:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner itself
                if global_path_s[index_closest_path_point] <= 10.0:
                    global_path_x.pop(index_closest_path_point)
                    global_path_y.pop(index_closest_path_point)
                    global_path_curvature_ok = False
                    break

        # also check if the curvature is smaller than the turning radius anywhere
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            if (curvature[i] * 2.0) > get_max_curvature(
                vehicle_params=vehicle_params, v=0.0
            )[0]:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner itself
                global_path_x.pop(index_closest_path_point)
                global_path_y.pop(index_closest_path_point)
                global_path_curvature_ok = False
                break

    # create the new global path
    new_global_path = np.array([np.array([global_path_x[0], global_path_y[0]])])
    for i in range(1, len(global_path_y)):
        new_global_path = np.concatenate(
            (
                new_global_path,
                np.array([np.array([global_path_x[i], global_path_y[i]])]),
            )
        )

    return new_global_path


def calc_frenet_trajectories(
    c_s: float,
    c_s_d: float,
    c_s_dd: float,
    c_d: float,
    c_d_d: float,
    c_d_dd: float,
    d_list: [float],
    t_list: [float],
    v_list: [float],
    dt: float,
    csp: CubicSpline2D,
    v_thr: float = 3.0,
    exec_timer=None,
):
    """
    Calculate all possible frenet trajectories from a given starting point and target lateral deviations, times and velocities.

    Args:
        c_s (float): Start longitudinal position.
        c_s_d (float): Start longitudinal velocity.
        c_s_dd (float): Start longitudinal acceleration
        c_d (float): Start lateral position.
        c_d_d (float): Start lateral velocity.
        c_d_dd (float): Start lateral acceleration.
        d_list ([float]): List of target lateral offsets to the reference spline.
        t_list ([float]): List of target end-times.
        v_list ([float]): List of target end-velocities.
        dt (float): Time step size of the trajectories.
        csp (CubicSpline2D): Reference spline of the global path.
        v_thr (float): Threshold velocity to distinguish slow and fast trajectories.
        exec_times_dict (dict): Dictionary for execution times. Defaults to None.

    Returns:
        [FrenetTrajectory]: List with all frenét trajectories.
    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer
    # list of all generated frenet trajectories
    fp_list = []
    # all end velocities
    for vT in v_list:
        if abs(c_s_d) < v_thr or abs(vT) < v_thr:
            lat_mode = "low_velocity"
        else:
            lat_mode = "high_velocity"
        # all end times
        for tT in t_list:
            # quartic polynomial in longitudinal direction
            with timer.time_with_cm(
                "simulation/calculate trajectories/initialize quartic polynomial"
            ):
                qp_long = quartic_polynomial(
                    xs=c_s, vxs=c_s_d, axs=c_s_dd, vxe=vT, axe=0.0, T=tT - dt
                )

            with timer.time_with_cm(
                "simulation/calculate trajectories/calculate quartic polynomial"
            ):
                # time vector
                t = list(np.arange(0.0, tT, dt))
                # longitudinal position and derivatives
                s = [qp_long.calc_point(t) for t in t]
                s_d = [qp_long.calc_first_derivative(t) for t in t]
                s_dd = [qp_long.calc_second_derivative(t) for t in t]
                s_ddd = [qp_long.calc_third_derivative(t) for t in t]

            s0 = s[0]
            # all lateral distances
            for dT in d_list:
                # quintic polynomial in lateral direction
                # for high velocities we have ds/dt and dd/dt
                ds = s[-1] - s0
                if lat_mode == "high_velocity":

                    with timer.time_with_cm(
                        "simulation/calculate trajectories/initialize quintic polynomial"
                    ):
                        qp_lat = quintic_polynomial(
                            xs=c_d,
                            vxs=c_d_d,
                            axs=c_d_dd,
                            xe=dT,
                            vxe=0.0,
                            axe=0.0,
                            T=tT - dt,
                        )

                    with timer.time_with_cm(
                        "simulation/calculate trajectories/calculate quintic polynomial"
                    ):
                        # lateral distance and derivatives
                        d = [qp_lat.calc_point(t) for t in t]
                        d_d = [qp_lat.calc_first_derivative(t) for t in t]
                        d_dd = [qp_lat.calc_second_derivative(t) for t in t]
                        d_ddd = [qp_lat.calc_third_derivative(t) for t in t]

                    d_d_time = d_d
                    d_dd_time = d_dd
                # for low velocities, we have ds/dt and dd/ds
                elif lat_mode == "low_velocity":
                    # singularity
                    if ds == 0:
                        ds = 0.00001

                    with timer.time_with_cm(
                        "simulation/calculate trajectories/initialize quintic polynomial"
                    ):
                        # the quintic polynomial shows dd/ds, so d(c_s)/ds and dd(c_s)/dds is needed
                        if c_s_d != 0.0:
                            c_d_d_not_time = c_d_d / c_s_d
                            c_d_dd_not_time = (c_d_dd - c_s_dd * c_d_d_not_time) / (
                                c_s_d ** 2
                            )
                        else:
                            c_d_d_not_time = 0.0
                            c_d_dd_not_time = 0.0

                        # Upper boundary for ds to avoid bad lat polynoms (solved by  if ds > abs(dT)?)
                        # ds = max(ds, 0.1)

                        qp_lat = quintic_polynomial(
                            xs=c_d,
                            vxs=c_d_d_not_time,
                            axs=c_d_dd_not_time,
                            xe=dT,
                            vxe=0.0,
                            axe=0.0,
                            T=ds,
                        )

                    with timer.time_with_cm(
                        "simulation/calculate trajectories/calculate quintic polynomial"
                    ):
                        # lateral distance and derivatives
                        d = [qp_lat.calc_point(s - s0) for s in s]
                        d_d = [qp_lat.calc_first_derivative(s - s0) for s in s]
                        d_dd = [qp_lat.calc_second_derivative(s - s0) for s in s]
                        d_ddd = [qp_lat.calc_third_derivative(s - s0) for s in s]

                    # since dd/ds, a conversion to dd/dt is needed
                    d_d_time = [s_d[i] * d_d[i] for i in range(len(d))]
                    d_dd_time = [
                        s_dd[i] * d_d[i] + (s_d[i] ** 2) * d_dd[i]
                        for i in range(len(d))
                    ]

                with timer.time_with_cm(
                    "simulation/calculate trajectories/calculate global trajectory/total"
                ):
                    # calculate global path with the cubic spline planner (input is s, ds/dt, dds/ddt and for high velocities, d, dd/dt, ddd/ddt is later converted to d, dd/ds, ddd/dds)
                    x, y, yaw, curv, v, a = calc_global_trajectory(
                        csp=csp,
                        s=s,
                        s_d=s_d,
                        s_dd=s_dd,
                        d=d,
                        d_d_lat=d_d,
                        d_dd_lat=d_dd,
                        lat_mode=lat_mode,
                        exec_timer=timer,
                    )

                with timer.time_with_cm(
                    "simulation/calculate trajectories/initialize trajectory"
                ):
                    # create frenet trajectory
                    fp = FrenetTrajectory(
                        t=t,
                        d=d,
                        d_d=d_d_time,
                        d_dd=d_dd_time,
                        d_ddd=d_ddd,
                        s=s,
                        s_d=s_d,
                        s_dd=s_dd,
                        s_ddd=s_ddd,
                        x=x,
                        y=y,
                        yaw=yaw,
                        v=v,
                        curv=curv,
                    )

                if ds > abs(dT):
                    fp_list.append(fp)

    return fp_list


def sort_frenet_trajectories(
    ego_state,
    fp_list: [FrenetTrajectory],
    global_path: np.ndarray,
    predictions: dict,
    mode: str,
    params: dict,
    planning_problem: PlanningProblem,
    scenario: Scenario,
    vehicle_params,
    ego_id: int,
    dt: float,
    sensor_radius: float,
    road_boundary,
    collision_checker,
    goal_area,
    exec_timer=None,
    reach_set=None,
):
    """Sort the frenet trajectories. Check validity of all frenet trajectories in fp_list and sort them by increasing cost.

    Args:
        ego_state (State): Current state of the ego vehicle.
        fp_list ([FrenetTrajectory]): List with all frenét trajectories.
        global_path (np.ndarray): Global path.
        predictions (dict): Predictions of the visible obstacles.
        mode (Str): Mode of the frenét planner.
        planning_problem (PlanningProblem): Planning problem of the scenario.
        scenario (Scenario): Scenario.
        vehicle_params (VehicleParameters): Parameters of the ego vehicle.
        ego_id (int): ID of the ego vehicle.
        dt (float): Delta time of the scenario.
        sensor_radius (float): Sensor radius for the sensor model.
        road_boundary (ShapeGroup): Shape group representing the road boundary.
        collision_checker (CollisionChecker): Collision checker for the scenario.
        goal_area (ShapeGroup): Shape group of the goal area.
        exec_times_dict (dict): Dictionary for the execution times. Defaults to None.

    Returns:
        [FrenetTrajectory]: List of sorted valid frenét trajectories.
        [FrenetTrajectory]: List of sorted invalid frenét trajectories.
        dict: Dictionary with execution times.
    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    validity_dict = {key: [] for key in VALIDITY_LEVELS}

    if mode == "ground_truth" or mode == "WaleNet":
        cost_predictions = None
    else:
        cost_predictions = predictions

    if predictions is not None:
        for fp in fp_list:

            fp.ego_risk_dict, fp.obst_risk_dict, fp.ego_harm_dict, fp.obst_harm_dict, fp.bd_harm = calc_risk(
                traj=fp,
                ego_state=ego_state,
                predictions=predictions,
                scenario=scenario,
                ego_id=ego_id,
                vehicle_params=vehicle_params,
                road_boundary=road_boundary,
                params=params,
                exec_timer=timer,
            )

    for fp in fp_list:
        with timer.time_with_cm("simulation/sort trajectories/check validity/total"):
            # check validity
            fp.valid_level, fp.reason_invalid = check_validity(
                ft=fp,
                ego_state=ego_state,
                scenario=scenario,
                vehicle_params=vehicle_params,
                risk_params=params['modes'],
                predictions=predictions,
                mode=mode,
                road_boundary=road_boundary,
                collision_checker=collision_checker,
                exec_timer=timer,
            )

            validity_dict[fp.valid_level].append(fp)

    validity_level = max(
        [lvl for lvl in VALIDITY_LEVELS if len(validity_dict[lvl])]
    )

    ft_list_highest_validity = validity_dict[validity_level]
    ft_list_invalid = [
        validity_dict[inv] for inv in VALIDITY_LEVELS if inv < validity_level
    ]
    ft_list_invalid = [item for sublist in ft_list_invalid for item in sublist]

    for fp in ft_list_highest_validity:
        (
            fp.cost,
            fp.cost_dict,
        ) = calc_trajectory_costs(
            traj=fp,
            global_path=global_path,
            ego_state=ego_state,
            validity_level=validity_level,
            planning_problem=planning_problem,
            params=params,
            scenario=scenario,
            ego_id=ego_id,
            dt=dt,
            predictions=cost_predictions,
            sensor_radius=sensor_radius,
            vehicle_params=vehicle_params,
            goal_area=goal_area,
            exec_timer=timer,
            mode=mode,
            reach_set=reach_set
        )

    return ft_list_highest_validity, ft_list_invalid, validity_dict


def calc_global_trajectory(
    csp: CubicSpline2D,
    s: [float],
    s_d: [float],
    s_dd: [float],
    d: [float],
    d_d_lat: [float],
    d_dd_lat: [float],
    lat_mode: str,
    exec_timer=None,
):
    """
    Calculate the global trajectory with a cubic spline reference and a frenet trajectory.

    Args:
        csp (CubicSpline2D): 2D cubic spline representing the global path.
        s ([float]): List with the values for the covered arc length of the spline.
        s_d ([float]): List with the values for the longitudinal velocity.
        s_dd ([float]): List with the values for the longitudinal acceleration.
        d ([float]): List with the values of the lateral offset from the spline.
        d_d_lat ([float]): List with the lateral velocity (defined over s or t, depending on lat_mode).
        d_dd_lat ([float]): List with the lateral acceleration (defined over s or t, depending on lat_mode)
        lat_mode (str): Determines if it is a high speed or low speed trajectory.
        exec_times_dict (dict): Dictionary with the execution times.

    Returns:
        [float]: x-position of the global trajectory.
        [float]: y-position of the global trajectory.
        [float]: Yaw angle of the global trajectory.
        [float]: Curvature of the global trajectory.
        [float]: Velocity of the global trajectory.
        [float]: Acceleration position of the global trajectory.
        dict: Dictionary with the execution times.
    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    x = []
    y = []
    yaw = []
    curv = []
    v = []
    a = []

    with timer.time_with_cm(
        "simulation/calculate trajectories/calculate global trajectory/convert to ds-dt"
    ):
        # if it is a high velocity frenét trajectory, dd/dt and ddd/ddt need to be converted to dd/ds and ddd/dds
        if lat_mode == "high_velocity":
            d_d = [d_d_lat[i] / s_d[i] for i in range(len(s))]
            d_dd = [
                (d_dd_lat[i] - d_d[i] * s_dd[i]) / (s_d[i] ** 2) for i in range(len(s))
            ]
        elif lat_mode == "low_velocity":
            d_d = d_d_lat
            d_dd = d_dd_lat

    with timer.time_with_cm(
        "simulation/calculate trajectories/calculate global trajectory/calculate reference points"
    ):
        # calculate the position of the reference path
        global_path_x = []
        global_path_y = []

        for si in s:
            global_path_x.append(csp.sx(si))
            global_path_y.append(csp.sy(si))

    with timer.time_with_cm(
        "simulation/calculate trajectories/calculate global trajectory/calculate reference gradients"
    ):
        # calculate derivations necessary to get the curvature
        dx = np.gradient(global_path_x, s)
        ddx = np.gradient(dx, s)
        dddx = np.gradient(ddx, s)
        dy = np.gradient(global_path_y, s)
        ddy = np.gradient(dy, s)
        dddy = np.gradient(ddy, s)

    with timer.time_with_cm(
        "simulation/calculate trajectories/calculate global trajectory/calculate reference yaw"
    ):
        # calculate yaw of the global path
        global_path_yaw = np.arctan2(dy, dx)

    with timer.time_with_cm(
        "simulation/calculate trajectories/calculate global trajectory/calculate reference curvature"
    ):
        # calculate the curvature of the global path
        global_path_curv = (np.multiply(dx, ddy) - np.multiply(ddx, dy)) / (
            np.power(dx, 2) + np.power(dy, 2) ** (3 / 2)
        )

    with timer.time_with_cm(
        "simulation/calculate trajectories/calculate global trajectory/calculate reference curvature derivation"
    ):
        # calculate the derivation of the global path's curvature
        z = np.multiply(dx, ddy) - np.multiply(ddx, dy)
        z_d = np.multiply(dx, dddy) - np.multiply(dddx, dy)
        n = (np.power(dx, 2) + np.power(dy, 2)) ** (3 / 2)
        n_d = (3 / 2) * np.multiply(
            np.power((np.power(dx, 2) + np.power(dy, 2)), 0.5),
            (2 * np.multiply(dx, ddx) + 2 * np.multiply(dy, ddy)),
        )
        global_path_curv_d = (np.multiply(z_d, n) - np.multiply(z, n_d)) / (
            np.power(n, 2)
        )

    with timer.time_with_cm(
        "simulation/calculate trajectories/calculate global trajectory/calculate trajectory states"
    ):
        # transform every point of the trajectory from the frenét frame to the global coordinate system
        for i in range(len(s)):
            # information from the global path necessary for the transformation
            curvature_si = global_path_curv[i]
            yaw_si = global_path_yaw[i]
            curvature_d_si = global_path_curv_d[i]
            pos_si = [global_path_x[i], global_path_y[i]]

            # transform yaw, position and velocity
            yaw_diff = math.atan(d_d[i] / (1 - curvature_si * d[i]))
            iyaw = yaw_diff + yaw_si
            sx, sy = pos_si
            ix = sx - d[i] * math.sin(yaw_si)
            iy = sy + d[i] * math.cos(yaw_si)
            iv = (s_d[i] * (1 - curvature_si * d[i])) / math.cos(yaw_diff)

            # transform curvature
            icurv = (
                (
                    (
                        d_dd[i]
                        + (curvature_d_si * d[i] + curvature_si * d_d[i])
                        * np.tan(yaw_diff)
                    )
                    * ((np.cos(yaw_diff) ** 2) / (1 - curvature_si * d[i]))
                )
                + curvature_si
            ) * (np.cos(yaw_diff) / (1 - curvature_si * d[i]))

            # transform acceleration
            ia = s_dd[i] * ((1 - curvature_si * d[i]) / np.cos(yaw_diff)) + (
                (s_d[i] ** 2) / (np.cos(yaw_diff))
            ) * (
                (1 - curvature_si * d[i])
                * np.tan(yaw_diff)
                * (
                    icurv * ((1 - curvature_si * d[i]) / np.cos(yaw_diff))
                    - curvature_si
                )
                - (curvature_d_si * d[i] + curvature_si * d_d[i])
            )

            x.append(ix)
            y.append(iy)
            yaw.append(iyaw)
            curv.append(icurv)
            v.append(iv)
            a.append(ia)

    return x, y, yaw, curv, v, a


def get_v_list(
    v_min: float,
    v_max: float,
    v_cur: float,
    v_goal_min: float = None,
    v_goal_max: float = None,
    n_samples: int = 3,
    mode: str = "linspace",
):
    """
    Get a list of end velocities for the frenét planner.

    Args:
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        v_cur (float): Velocity at the current state.
        v_goal_min (float): Minimum goal velocity. Defaults to None.
        v_goal_max (float): Maximum goal velocity. Defaults to None.
        n_samples (int): Number of desired velocities. Defaults to 3.
        mode (Str): Chosen mode. (linspace, deterministic, or random).
            Defaults to linspace.

    Returns:
        list(float): A list of velocities.

    """
    # modes: 0 = linspace, 1 = deterministic, 2 = random

    # check if n_samples is valid
    if n_samples <= 0:
        raise ValueError("Number of samples must be at least 1")

    # check if both goal velocities are None
    if (v_goal_min is None or v_goal_max is None) and v_goal_max != v_goal_min:
        raise AttributeError("Both goal velocities must be None or both must be filled")

    if mode not in ["linspace", "deterministic", "random"]:
        raise ValueError("V-list mode must be linspace, deterministic, or random")

    # return for the linspace mode
    if mode == "linspace":
        if v_goal_min is None:
            return np.linspace(v_min, v_max, n_samples)
        else:
            return np.linspace(
                max(min(v_goal_min, v_min), 0.001), max(v_goal_max, v_max), n_samples
            )

    # check if n_samples is valid for the chosen mode
    if mode == "deterministic":
        if v_goal_min is None:
            if n_samples > 10:
                raise ValueError(
                    "n_samples can not be greater than 10 in deterministic mode"
                )
        else:
            if n_samples > 10:
                raise ValueError(
                    "n_samples can not be greater than 11 in deterministic mode"
                )

    # calculate mean of the goal velocities
    if v_goal_min is not None:
        v_goal = v_goal_min + ((v_goal_max - v_goal_min) / 2)

    # create the order in which the velocities should be appended
    if v_goal_min is None:
        v_append_list = [v_cur, v_min, v_max]
    else:
        v_append_list = [v_goal, v_min, v_max, v_cur]

    v_list = []

    # add the velocities from the append list, further points are added from the density distribution
    for i in range(n_samples):
        if i < len(v_append_list):
            v_list.append(v_append_list[i])
        else:
            break

    # calculate the density distribution if necessary
    if n_samples <= len(v_append_list):
        v_list.sort()
        return v_list
    else:
        n_remaining_samples = n_samples - len(v_append_list)
        # loc is the lower limit, scale the upper limit
        loc = min(v_min, v_goal_min)
        scale = max(v_max, v_goal_max)
        # get the beta-distribution (depends on the available velocities)
        if v_goal_min is not None:
            a, b, floc, fscale = beta.fit(
                floc=loc,
                fscale=scale,
                data=[
                    loc + ((v_goal - loc) / 2),
                    v_goal,
                    scale - ((scale - v_goal) / 2),
                ],
            )
        else:
            a, b, floc, fscale = beta.fit(
                floc=loc,
                fscale=scale,
                data=[loc + ((v_cur - loc) / 2), v_cur, scale - ((scale - v_cur) / 2)],
            )
        median = beta.median(a=a, b=b, loc=loc, scale=scale)

        # for the deterministic mode, add the following quartiles
        if mode == "deterministic":
            alpha_even = [0.2, 0.4, 0.6, 0.8]
            alpha_odd = [0.3, 0.5, 0.7, 0.9]

            # for an odd number of samples, add the median and the quartile from alpha_odd
            if n_remaining_samples % 2 == 1:
                v_list.append(median)
                alpha = alpha_odd
            # for an even number of samples, add the quartile from alpha_even
            else:
                n_remaining_samples += 1
                alpha = alpha_even
            for i in range(1, int(n_remaining_samples / 2) + 1):
                interv = beta.interval(alpha=alpha[i], a=a, b=b, loc=loc, scale=scale)
                v_list.append(interv[0])
                v_list.append(interv[1])

        # for the random mode, just add random velocities from the beta distribution
        elif mode == "random":
            random_samples = beta.rvs(
                a=a, b=b, loc=loc, scale=scale, size=n_remaining_samples
            )
            v_list = [*v_list, *random_samples]

    v_list.sort()

    return v_list
