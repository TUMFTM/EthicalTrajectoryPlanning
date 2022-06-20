"""Functions to get vehicle properties or geometrical parameters."""

from commonroad.scenario.obstacle import ObstacleType
import numpy as np
from EthicalTrajectoryPlanning.risk_assessment.helpers.collision_helper_function import (
    angle_range,
    create_tvobstacle,
)
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_object,
)


def get_obstacle_mass(obstacle_type, size):
    """
    Get the mass of the considered obstacle.

    Args:
        obstacle_type (ObstacleType): Type of considered obstacle.
        size (float): Size (length * width) of the vehicle in m².

    Returns:
        Mass (float): Estimated mass of considered obstacle.
    """
    mass_mapping = {
        ObstacleType.CAR: -1333.5 + 526.9 * np.power(size, 0.8),
        ObstacleType.TRUCK: 25000,
        ObstacleType.BUS: 13000,
        ObstacleType.BICYCLE: 90,
        ObstacleType.PEDESTRIAN: 75,
        ObstacleType.PRIORITY_VEHICLE: -1333.5 + 526.9 * np.power(size, 0.8),
        ObstacleType.PARKED_VEHICLE: -1333.5 + 526.9 * np.power(size, 0.8),
        ObstacleType.TRAIN: 118800,
        ObstacleType.MOTORCYCLE: 250,
        ObstacleType.TAXI: -1333.5 + 526.9 * np.power(size, 0.8),
        ObstacleType.UNKNOWN: 100,
    }

    return mass_mapping[obstacle_type]


def calc_delta_v(vehicle_1, vehicle_2, pdof):
    """
    Calculate the difference between pre-crash and post-crash speed.

    Args:
        vehicle_1 (HarmParameters): dictionary with crash relevant parameters
            for the first vehicle
        vehicle_2 (HarmParameters): dictionary with crash relevant parameters
            for the second vehicle
        pdof (float): crash angle [rad].

    Returns:
        float: Delta v for the first vehicle
        float: Delta v for the second vehicle
    """
    delta_v = np.sqrt(
        np.power(vehicle_1.velocity, 2)
        + np.power(vehicle_2.velocity, 2)
        + 2 * vehicle_1.velocity * vehicle_2.velocity * np.cos(pdof)
    )

    veh_1_delta_v = vehicle_2.mass / (vehicle_1.mass + vehicle_2.mass) * delta_v
    veh_2_delta_v = vehicle_1.mass / (vehicle_1.mass + vehicle_2.mass) * delta_v

    return veh_1_delta_v, veh_2_delta_v


def calc_crash_angle(traj, predictions, scenario, obstacle_id, modes, vehicle_params):
    """
    Calculate the PDOF between ego vehicle and obstacle.

    Calculate the PDOF if the considered Frenét trajectory and the predicted
    obstacle trajectory intersect each other. Constant within the considered
    Frenét trajectory.

    Args:
        traj (FrenetTrajectory): Considered Frenét trajectory.
        predictions (dict): Predictions for the visible obstacles.
        scenario (Scenario): Considered scenario.
        obstacle_id (int): ID of the currently considered obstacle.
        modes (Dict): Risk modes. Read from risk.json.
        vehicle_params (VehicleParameters): Vehicle parameters of the
            considered vehicle.

    Returns:
        float: Principle degree of force.
        float: Crash angle for the ego vehicle [rad].
        float: Crash angle for the obstacle [rad].
    """
    crash_time = None

    # get collision time
    for i in range(len(traj.t)):
        # create collision object and check for collision
        current_state_collision_object = create_tvobstacle(
            traj_list=[
                [
                    traj.x[i],
                    traj.y[i],
                    traj.yaw[i],
                ]
            ],
            box_length=vehicle_params.l / 2,
            box_width=vehicle_params.w / 2,
            start_time_step=i,
        )

        co = create_collision_object(scenario.obstacle_by_id(obstacle_id=obstacle_id))

        if current_state_collision_object.collide(co):
            crash_time = i
            break

    # get state at crash time
    if crash_time is not None:
        pdof = (
            traj.yaw[crash_time]
            - predictions[obstacle_id]["orientation_list"][crash_time]
            + 180
        )
        pos_diff = [
            predictions[obstacle_id]["pos_list"][crash_time][0] - traj.x[crash_time],
            predictions[obstacle_id]["pos_list"][crash_time][1] - traj.y[crash_time],
        ]
        rel_angle = np.arctan2(pos_diff[1], pos_diff[0])
        ego_angle = rel_angle - traj.yaw[crash_time]
        obs_angle = (
            np.pi + rel_angle - predictions[obstacle_id]["orientation_list"][crash_time]
        )

    else:
        pdof, ego_angle, obs_angle = estimate_crash_angle(
            traj=traj, predictions=predictions, obstacle_id=obstacle_id, modes=modes
        )

    pdof = angle_range(pdof)
    ego_angle = angle_range(ego_angle)
    obs_angle = angle_range(obs_angle)

    return pdof, ego_angle, obs_angle


def estimate_crash_angle(traj, predictions, obstacle_id, modes):
    """
    Estimate the PDOF if not calculable.

    Estimate the crash angle between ego vehicle and obstacle, if not
    the predicted trajectoies do not intersect each other. Constant within the
    considered Frenét trajectory.

    Args:
        traj (FrenetTrajectory): Considered frenét trajectory.
        predictions (dict): Predictions for the visible obstacles.
        obstacle_id (int): ID of the currently considered obstacle.
        modes (Dict): Risk modes. Read from risk.json.

    Returns:
        float: Principle degree of force.
        float: Crash angle for the ego vehicle [rad].
        float: Crash angle for the obstacle [rad].
    """
    # extract relevant variables from predictions
    pred_path = predictions[obstacle_id]['pos_list'][0]
    pred_length = len(traj.t)
    pred_v = predictions[obstacle_id]['v_list'][0]
    pred_yaw = predictions[obstacle_id]['orientation_list'][0]

    # trafo matrix from obstacle vehicle system to xy-system
    T_xy_cur = np.ndarray(shape=(2, 2))
    T_xy_cur[0][0] = np.cos(pred_yaw)
    T_xy_cur[0][1] = -np.sin(pred_yaw)
    T_xy_cur[1][0] = np.sin(pred_yaw)
    T_xy_cur[1][1] = np.cos(pred_yaw)

    # trafo matrix from xy-system to obstacle system
    T_cur_xy = np.ndarray(shape=(2, 2))
    T_cur_xy[0][0] = np.cos(pred_yaw)
    T_cur_xy[0][1] = np.sin(pred_yaw)
    T_cur_xy[1][0] = -np.sin(pred_yaw)
    T_cur_xy[1][1] = np.cos(pred_yaw)

    # ego position
    ego_pos = np.array([[traj.x[0]], [traj.y[0]]])

    # obstacle position
    obs_pos = np.array([[pred_path[0]], [pred_path[1]]])

    # difference between ego position and obstacle position in obstacle frame
    diff = ego_pos - obs_pos
    diff_obs = np.matmul(T_cur_xy, diff)

    # lateral acceleration
    ay = modes["lateral_acceleration"] * 9.81

    # calc radius and turn rate
    turn_rate = ay / np.power(pred_v, 2)
    radius = pred_v / turn_rate

    # create list to store position differences for different crash-initiating
    # trajectories
    delta_angle = []
    curve = None
    trajectory = {}
    trajectory_list = []

    # ego vehicle is on left side of obstacle
    if diff_obs[1] > 0:
        left_side = 1
    else:
        left_side = -1

    # ego vehicle and obstacle heading in same direction
    if -np.pi / 2 < traj.yaw[0] - pred_yaw < np.pi / 2:
        same_direction = True
    else:
        same_direction = False

    # calculate number of sections
    num = int(90 / modes["crash_angle_accuracy"] + 1)

    # iterate through angle range [0°;90°]
    gamma = np.linspace(0, 0.5 * np.pi, num=num)
    for angle in gamma:
        path = []
        x = []
        y = []
        delta_time = []
        timestep = range(pred_length)
        # add position for every timestep
        for ts in timestep:
            # curve part of crash trajectory
            if ts <= angle / turn_rate:
                curve = [
                    [radius * np.sin(turn_rate * ts)],
                    [left_side * (-radius * np.cos(turn_rate * ts) + radius)],
                ]
                cur = np.matmul(T_xy_cur, curve)
                cur = [cur[0][0], cur[1][0]]
                cur[0] += pred_path[0]
                cur[1] += pred_path[1]
                path.append(cur)
            else:
                # trafo matrix from after curve to before curve
                T_cur_str = np.ndarray(shape=(2, 2))
                T_cur_str[0][0] = np.cos(angle)
                T_cur_str[0][1] = left_side * (-np.sin(angle))
                T_cur_str[1][0] = left_side * np.sin(angle)
                T_cur_str[1][1] = np.cos(angle)

                # linear part of crash trajectory
                linear = [[pred_v * (ts - angle / turn_rate)], [0]]
                if curve is None:
                    curve = np.array([[0], [0]])
                cur = np.matmul(T_xy_cur, curve) + np.matmul(
                    T_xy_cur, np.matmul(T_cur_str, linear)
                )
                cur = [cur[0][0], cur[1][0]]
                cur[0] += pred_path[0]
                cur[1] += pred_path[1]
                path.append(cur)

        # calculate difference between ego vehicle position and assumed crash
        # trajectories
        for ts in range(pred_length):
            x.append(path[ts][0])
            y.append(path[ts][1])
            delta_time.append(np.linalg.norm([traj.x[ts] - x[ts], traj.y[ts] - y[ts]]))

        # add trajectory to dict
        trajectory["x"] = x
        trajectory["y"] = y

        delta_angle.append(delta_time)
        trajectory_list.append(trajectory)

    # calculate minimum distance for all possible collision trajectories
    delta_min = np.where(delta_angle == np.amin(delta_angle))

    # get iteration step of angle with minimum distance
    angle_iter = delta_min[0][0]
    crash_time = delta_min[1][0]

    # get pdof
    # differentiate between four cases
    if same_direction is True and left_side == 1:
        # obstacle coming from behind and turning left [0;90°]
        angle = 0.5 * np.pi / (num - 1) * angle_iter
    elif same_direction is True and left_side == -1:
        # obstacle coming from behind and turning right[-90°;0]
        angle = -0.5 * np.pi / (num - 1) * angle_iter
    elif same_direction is False and left_side == 1:
        # obstacle coming towards ego vehicle and turning left [-180°;-90°]
        angle = 0.5 * np.pi / (num - 1) * angle_iter
    else:
        # obstacle coming towards ego vehicle and turning right [90°;180°]
        angle = -0.5 * np.pi / (num - 1) * angle_iter

    if crash_time < abs(angle) / turn_rate:
        rel_angle = crash_time * turn_rate * angle / abs(angle)
    else:
        rel_angle = angle

    # get ego and obstacle angle
    pos_diff = [
        trajectory_list[angle_iter]["x"][crash_time] - traj.x[crash_time],
        trajectory_list[angle_iter]["y"][crash_time] - traj.y[crash_time],
    ]
    rel_angle2 = np.arctan2(pos_diff[1], pos_diff[0])

    # get absolute crash angle
    pdof = pred_yaw - traj.yaw[crash_time] + rel_angle + np.pi
    ego_angle = rel_angle2 - traj.yaw[crash_time]
    obs_angle = (
        np.pi + rel_angle2 - predictions[obstacle_id]["orientation_list"][crash_time]
    )

    pdof = angle_range(pdof)
    ego_angle = angle_range(ego_angle)
    obs_angle = angle_range(obs_angle)

    return pdof, ego_angle, obs_angle


def calc_crash_angle_simple(traj, predictions, obstacle_id, time_step):
    """
    Simplified PDOF based on vehicle orientation.

    Calculate the crash angle between the ego vehicle and the obstacle based
    on a simple approximation by considering the current orientation. Variant
    over time for a considered Frenét trajectory.

    Args:
        traj (FrenetTrajectory): Considered frenét trajectory.
        predictions (dict): Predictions for the visible obstacles.
        obstacle_id (int): ID of the currently considered obstacle.
        time_step (Int): Currently considered time step.

    Returns:
        float: Estimated crash angle [rad].
    """
    pdof = (
        predictions[obstacle_id]["orientation_list"][time_step]
        - traj.yaw[time_step]
        + np.pi
    )
    pos_diff = [
        predictions[obstacle_id]["pos_list"][time_step][0] - traj.x[time_step],
        predictions[obstacle_id]["pos_list"][time_step][1] - traj.y[time_step],
    ]
    rel_angle = np.arctan2(pos_diff[1], pos_diff[0])
    ego_angle = rel_angle - traj.yaw[time_step]
    obs_angle = (
        np.pi + rel_angle - predictions[obstacle_id]["orientation_list"][time_step]
    )

    return pdof, ego_angle, obs_angle
