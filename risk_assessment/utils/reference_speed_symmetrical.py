
"""Reference speed harm actual functions for symmetrical models."""

import numpy as np


def get_protected_inj_prob_ref_speed_complete_sym(velocity,
                                                  angle,
                                                  coeff):
    """
    RS12S.

    Get the injury probability via the reference speed model for 12
    considered impact areas. Area coefficients are set symmetrically.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # get angle coefficient
    if -15 / 180 * np.pi < angle < 15 / 180 * np.pi:  # impact 12
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_12"]
    elif 15 / 180 * np.pi <= angle < 45 / 180 * np.pi:  # impact 11
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_1_11"]
    elif -15 / 180 * np.pi >= angle > -45 / 180 * np.pi:  # impact 1
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_1_11"]
    elif 45 / 180 * np.pi <= angle < 75 / 180 * np.pi:  # impact 10
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_2_10"]
    elif -45 / 180 * np.pi >= angle > -75 / 180 * np.pi:  # impact 2
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_2_10"]
    elif 75 / 180 * np.pi <= angle < 105 / 180 * np.pi:  # impact 9
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_3_9"]
    elif -75 / 180 * np.pi >= angle > -105 / 180 * np.pi:  # impact 3
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_3_9"]
    elif 105 / 180 * np.pi <= angle < 135 / 180 * np.pi:  # impact 8
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_4_8"]
    elif -105 / 180 * np.pi >= angle > -135 / 180 * np.pi:  # impact 4
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_4_8"]
    elif 135 / 180 * np.pi <= angle < 165 / 180 * np.pi:  # impact 7
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_5_7"]
    elif -135 / 180 * np.pi >= angle > -165 / 180 * np.pi:  # impact 5
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_5_7"]
    else:  # impact 6
        reference = \
            coeff["ref_speed"]["complete_sym_angle_areas"]["ref_speed_6"]

    # model
    if velocity < reference:
        p_mais = \
            np.power(velocity / reference,
                     coeff["ref_speed"]["complete_sym_angle_areas"]["exp"])
    else:
        p_mais = 1

    return p_mais


def get_protected_inj_prob_ref_speed_reduced_sym(velocity,
                                                 angle,
                                                 coeff):
    """
    RS4S.

    Get the injury probability via the reference speed model for 4
    considered impact areas. Area coefficients are set symmetrically.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # get angle coefficient
    if -45 / 180 * np.pi < angle < 45 / 180 * np.pi:  # front crash
        reference = \
            coeff["ref_speed"]["reduced_sym_angle_areas"]["ref_speed_front"]
    elif 45 / 180 * np.pi <= angle < 135 / 180 * np.pi:  # driver-side crash
        reference = \
            coeff["ref_speed"]["reduced_sym_angle_areas"]["ref_speed_side"]
    elif -45 / 180 * np.pi >= angle > -135 / 180 * np.pi:  # right-side crash
        reference = \
            coeff["ref_speed"]["reduced_sym_angle_areas"]["ref_speed_side"]
    else:  # front crash
        reference = \
            coeff["ref_speed"]["reduced_sym_angle_areas"]["ref_speed_rear"]

    # model
    if velocity < reference:
        p_mais = \
            np.power(velocity / reference,
                     coeff["ref_speed"]["reduced_sym_angle_areas"]["exp"])
    else:
        p_mais = 1

    return p_mais


def get_protected_inj_prob_ref_speed_ignore_angle(velocity,
                                                  coeff):
    """
    RS1S.

    Get the injury probability via the reference speed model. Impact areas
    are not considered.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # model
    if velocity < coeff["ref_speed"]["ignore_angle"]["ref_speed"]:
        p_mais = \
            np.power(velocity /
                     coeff["ref_speed"]["ignore_angle"]["ref_speed"],
                     coeff["ref_speed"]["ignore_angle"]["exp"])
    else:
        p_mais = 1

    return p_mais
