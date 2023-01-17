
"""Logistic regression harm actual functions for asymmetrical models."""

import numpy as np


def get_protected_inj_prob_log_reg_complete(velocity,
                                            angle,
                                            coeff):
    """
    LR12A.

    Get the injury probability via logistic regression for 12 considered
    impact areas.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # get angle coefficient
    for i in range(len(angle)):
        if -15 / 180 * np.pi < angle[i] < 15 / 180 * np.pi:  # impact 12
            angle[i] = 0
        elif 15 / 180 * np.pi <= angle[i] < 45 / 180 * np.pi:  # impact 11
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_11"]
        elif -15 / 180 * np.pi >= angle[i] > -45 / 180 * np.pi:  # impact 1
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_1"]
        elif 45 / 180 * np.pi <= angle[i] < 75 / 180 * np.pi:  # impact 10
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_10"]
        elif -45 / 180 * np.pi >= angle[i] > -75 / 180 * np.pi:  # impact 2
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_2"]
        elif 75 / 180 * np.pi <= angle[i] < 105 / 180 * np.pi:  # impact 9
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_9"]
        elif -75 / 180 * np.pi >= angle[i] > -105 / 180 * np.pi:  # impact 3
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_3"]
        elif 105 / 180 * np.pi <= angle[i] < 135 / 180 * np.pi:  # impact 8
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_8"]
        elif -105 / 180 * np.pi >= angle[i] > -135 / 180 * np.pi:  # impact 4
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_4"]
        elif 135 / 180 * np.pi <= angle[i] < 165 / 180 * np.pi:  # impact 7
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_7"]
        elif -135 / 180 * np.pi >= angle[i] > -165 / 180 * np.pi:  # impact 5
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_5"]
        else:  # impact 6
            angle[i] = coeff["log_reg"]["complete_angle_areas"]["Imp_6"]

    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["complete_angle_areas"]
                             ["const"] - coeff["log_reg"]
                             ["complete_angle_areas"]["speed"] * velocity -
                             angle))

    return p_mais


def get_protected_inj_prob_log_reg_reduced(velocity,
                                           angle,
                                           coeff):
    """
    LR4A.

    Get the injury probability via logistic regression for 4 considered
    impact areas.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # get angle coefficient
    for i in range(len(angle)):
        if -45 / 180 * np.pi < angle[i] < 45 / 180 * np.pi:  # front crash
            angle[i] = 0
        elif 45 / 180 * np.pi <= angle[i] < 135 / 180 * np.pi:  # driver-side crash
            angle[i] = coeff["log_reg"]["reduced_angle_areas"]["driver_side"]
        elif -45 / 180 * np.pi >= angle[i] > -135 / 180 * np.pi:  # right-side crash
            angle[i] = coeff["log_reg"]["reduced_angle_areas"]["right_side"]
        else:  # rear crash
            angle[i] = coeff["log_reg"]["reduced_angle_areas"]["rear"]

    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["reduced_angle_areas"]
                             ["const"] - coeff["log_reg"]
                             ["reduced_angle_areas"]["speed"] * velocity -
                             angle))

    return p_mais
