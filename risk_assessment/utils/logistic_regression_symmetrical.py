
"""Logistic regression harm actual functions for symmetrical models."""

import numpy as np


def get_protected_inj_prob_log_reg_complete_sym(velocity,
                                                angle,
                                                coeff):
    """
    LR12S.

    Get the injury probability via logistic regression for 12 considered
    impact areas. Area coefficients are set symmetrically.

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
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_1_11"]
        elif -15 / 180 * np.pi >= angle[i] > -45 / 180 * np.pi:  # impact 1
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_1_11"]
        elif 45 / 180 * np.pi <= angle[i] < 75 / 180 * np.pi:  # impact 10
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_2_10"]
        elif -45 / 180 * np.pi >= angle[i] > -75 / 180 * np.pi:  # impact 2
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_2_10"]
        elif 75 / 180 * np.pi <= angle[i] < 105 / 180 * np.pi:  # impact 9
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_3_9"]
        elif -75 / 180 * np.pi >= angle[i] > -105 / 180 * np.pi:  # impact 3
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_3_9"]
        elif 105 / 180 * np.pi <= angle[i] < 135 / 180 * np.pi:  # impact 8
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_4_8"]
        elif -105 / 180 * np.pi >= angle[i] > -135 / 180 * np.pi:  # impact 4
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_4_8"]
        elif 135 / 180 * np.pi <= angle[i] < 165 / 180 * np.pi:  # impact 7
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_5_7"]
        elif -135 / 180 * np.pi >= angle[i] > -165 / 180 * np.pi:  # impact 5
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_5_7"]
        else:  # impact 6
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_6"]

    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["complete_sym_angle_areas"]
                             ["const"] - coeff["log_reg"]
                             ["complete_sym_angle_areas"]["speed"] * velocity -
                             angle))

    return p_mais


def get_protected_inj_prob_log_reg_reduced_sym(velocity,
                                               angle,
                                               coeff):
    """
    LR4S.

    Get the injury probability via logistic regression for 4 considered
    impact areas. Area coefficients are set symmetrically.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # get angle coefficient
    t_a = 45 / 180 * np.pi
    t_b = 3 * t_a
    unpack = False
    if isinstance(angle, float):
        angle = [angle]
        unpack = True
    for i in range(len(angle)):
        if -t_a < angle[i] < t_a:  # front crash
            angle[i] = 0
        elif t_a <= angle[i] < t_b:  # driver-side crash
            angle[i] = coeff["log_reg"]["reduced_sym_angle_areas"]["side"]
        elif -t_a >= angle[i] > -t_b:  # right-side crash
            angle[i] = coeff["log_reg"]["reduced_sym_angle_areas"]["side"]
        else:  # rear crash
            angle[i] = coeff["log_reg"]["reduced_sym_angle_areas"]["rear"]

    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["reduced_sym_angle_areas"]
                             ["const"] - coeff["log_reg"]
                             ["reduced_sym_angle_areas"]["speed"] * velocity -
                             angle))
    if unpack:
        p_mais = p_mais[0]

    return p_mais


# change add a parameter
def get_protected_inj_prob_log_reg_ignore_angle(velocity,
                                                coeff,
                                                angle=0):
    """
    LR1S.

    Get the injury probability via logistic regression. Impact areas are not
    considered.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["ignore_angle"]["const"] -
                             coeff["log_reg"]["ignore_angle"]["speed"] *
                             velocity))

    return p_mais
