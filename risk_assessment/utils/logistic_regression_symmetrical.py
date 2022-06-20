
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
    if -15 / 180 * np.pi < angle < 15 / 180 * np.pi:  # impact 12
        area = 0
    elif 15 / 180 * np.pi <= angle < 45 / 180 * np.pi:  # impact 11
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_1_11"]
    elif -15 / 180 * np.pi >= angle > -45 / 180 * np.pi:  # impact 1
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_1_11"]
    elif 45 / 180 * np.pi <= angle < 75 / 180 * np.pi:  # impact 10
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_2_10"]
    elif -45 / 180 * np.pi >= angle > -75 / 180 * np.pi:  # impact 2
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_2_10"]
    elif 75 / 180 * np.pi <= angle < 105 / 180 * np.pi:  # impact 9
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_3_9"]
    elif -75 / 180 * np.pi >= angle > -105 / 180 * np.pi:  # impact 3
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_3_9"]
    elif 105 / 180 * np.pi <= angle < 135 / 180 * np.pi:  # impact 8
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_4_8"]
    elif -105 / 180 * np.pi >= angle > -135 / 180 * np.pi:  # impact 4
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_4_8"]
    elif 135 / 180 * np.pi <= angle < 165 / 180 * np.pi:  # impact 7
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_5_7"]
    elif -135 / 180 * np.pi >= angle > -165 / 180 * np.pi:  # impact 5
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_5_7"]
    else:  # impact 6
        area = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_6"]

    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["complete_sym_angle_areas"]
                             ["const"] - coeff["log_reg"]
                             ["complete_sym_angle_areas"]["speed"] * velocity -
                             area))

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
    if -45 / 180 * np.pi < angle < 45 / 180 * np.pi:  # front crash
        area = 0
    elif 45 / 180 * np.pi <= angle < 135 / 180 * np.pi:  # driver-side crash
        area = coeff["log_reg"]["reduced_sym_angle_areas"]["side"]
    elif -45 / 180 * np.pi >= angle > -135 / 180 * np.pi:  # right-side crash
        area = coeff["log_reg"]["reduced_sym_angle_areas"]["side"]
    else:  # rear crash
        area = coeff["log_reg"]["reduced_sym_angle_areas"]["rear"]

    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["reduced_sym_angle_areas"]
                             ["const"] - coeff["log_reg"]
                             ["reduced_sym_angle_areas"]["speed"] * velocity -
                             area))

    return p_mais


def get_protected_inj_prob_log_reg_ignore_angle(velocity,
                                                coeff):
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
