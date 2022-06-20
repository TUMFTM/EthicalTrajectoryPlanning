"""Helper functions for collision probability."""

import numpy as np


def distance(pos1: np.array, pos2: np.array):
    """
    Return the euclidean distance between 2 points.

    Args:
        pos1 (np.array): First point.
        pos2 (np.array): Second point.

    Returns:
        float: Distance between point 1 and point 2.
    """
    return np.sqrt((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) +
                   (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]))


def get_unit_vector(angle: float):
    """
    Get the unit vector for a given angle.

    Args:
        angle (float): Considered angle.

    Returns:
        float: Unit vector of the considered angle
    """
    return np.array([np.cos(angle), np.sin(angle)])
