
"""Reference speed harm actual functions for asymmetrical models."""

import numpy as np


def get_protected_inj_prob_ref_speed_complete(velocity,
                                              angle,
                                              coeff):
    """
    RS12A.

    Get the injury probability via the reference speed model for 12
    considered impact areas.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # get angle coefficient
    reference = np.zeros_like(angle)
    for i in range(len(angle)):
        if -15 / 180 * np.pi < angle[i] < 15 / 180 * np.pi:  # impact 12
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_12"]
        elif 15 / 180 * np.pi <= angle[i] < 45 / 180 * np.pi:  # impact 11
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_11"]
        elif -15 / 180 * np.pi >= angle[i] > -45 / 180 * np.pi:  # impact 1
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_1"]
        elif 45 / 180 * np.pi <= angle[i] < 75 / 180 * np.pi:  # impact 10
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_10"]
        elif -45 / 180 * np.pi >= angle[i] > -75 / 180 * np.pi:  # impact 2
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_2"]
        elif 75 / 180 * np.pi <= angle[i] < 105 / 180 * np.pi:  # impact 9
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_9"]
        elif -75 / 180 * np.pi >= angle[i] > -105 / 180 * np.pi:  # impact 3
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_3"]
        elif 105 / 180 * np.pi <= angle[i] < 135 / 180 * np.pi:  # impact 8
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_8"]
        elif -105 / 180 * np.pi >= angle[i] > -135 / 180 * np.pi:  # impact 4
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_4"]
        elif 135 / 180 * np.pi <= angle[i] < 165 / 180 * np.pi:  # impact 7
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_7"]
        elif -135 / 180 * np.pi >= angle[i] > -165 / 180 * np.pi:  # impact 5
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_5"]
        else:  # impact 6
            reference = coeff["ref_speed"]["complete_angle_areas"]["ref_speed_6"]

    temp = np.power(velocity / reference,
                          coeff["ref_speed"]["complete_angle_areas"]["exp"])

    # model
    p_mais = np.zeros_like(angle)
    for i in range(len(angle)):
        if velocity[i] < reference[i]:
            p_mais[i] = temp[i]
        else:
            p_mais[i] = 1

    return p_mais


def get_protected_inj_prob_ref_speed_reduced(velocity,
                                             angle,
                                             coeff):
    """
    RS4A.

    Get the injury probability via the reference speed model for 4
    considered impact areas.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # get angle coefficient
    reference = np.zeros_like(angle)
    for i in range(len(angle)):
        if -45 / 180 * np.pi < angle[i] < 45 / 180 * np.pi:  # front crash
            reference = \
                coeff["ref_speed"]["reduced_angle_areas"]["ref_speed_front"]
        elif 45 / 180 * np.pi <= angle[i] < 135 / 180 * np.pi:  # driver-side crash
            reference = \
                coeff["ref_speed"]["reduced_angle_areas"]["ref_speed_driver_side"]
        elif -45 / 180 * np.pi >= angle[i] > -135 / 180 * np.pi:  # right-side crash
            reference = \
                coeff["ref_speed"]["reduced_angle_areas"]["ref_speed_right_side"]
        else:  # rear crash
            reference = \
                coeff["ref_speed"]["reduced_angle_areas"]["ref_speed_rear"]

    # model
    temp = np.power(velocity / reference,
                          coeff["ref_speed"]["reduced_angle_areas"]["exp"])
    p_mais = np.zeros_like(angle)
    for i in range(len(angle)):
        if velocity[i] < reference[i]:
            p_mais[i] = temp[i]
        else:
            p_mais[i] = 1

    return p_mais
