"""Helper functions for mod risk."""

import commonroad_dc.pycrcc as pycrcc
import numpy as np


def create_tvobstacle(traj_list: [[float]],
                      box_length: float,
                      box_width: float,
                      start_time_step: int):
    """
    Return a time variant collision object.

    Args:
        traj_list ([[float]]): List with the trajectory
            ([x-position, y-position, orientation]).
        box_length (float): Length of the obstacle.
        box_width (float): Width of the obstacle.
        start_time_step (int): Time step of the initial state.

    Returns:
        pyrcc.TimeVariantCollisionObject: Collision object.
    """
    # time variant object starts at the given time step
    tv_obstacle = pycrcc.TimeVariantCollisionObject(
        time_start_idx=start_time_step)
    for state in traj_list:
        # append each state to the time variant collision object
        tv_obstacle.append_obstacle(
            pycrcc.RectOBB(box_length, box_width, state[2], state[0], state[1])
        )
    return tv_obstacle


def angle_range(angle):
    """
    Return an angle in [rad] in the interval ]-pi; pi].

    Args:
        angle (float): Angle in rad.
    Returns:
        float: angle in rad in the interval ]-pi; pi]

    """
    while angle <= -np.pi or angle > np.pi:
        if angle <= -np.pi:
            angle += 2 * np.pi
        elif angle > np.pi:
            angle -= 2 * np.pi

    return angle
