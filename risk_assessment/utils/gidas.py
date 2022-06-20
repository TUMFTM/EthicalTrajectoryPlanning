"""
Harm estimation model published by GIDAS.

Get harm via the logistic regression published by Bosch GmbH at the 20th
anniversary symposium of GIDAS (german in-depth accident study).
HARM IS CALCULATED AS P(MAIS 2+)!!!

"""

import numpy as np
from EthicalTrajectoryPlanning.risk_assessment.helpers.properties import calc_delta_v


def get_protected_gidas_harm(ego_vehicle, obstacle, pdof: float, coeff):
    """
    GIDAS harm model.

    Get the harm for two possible collision partners, if both are vehicles
    with a protective crash structure, via the logistic regression
    published by Bosch at the GIDAS symposia 2019.

    Args:
        ego_vehicle (HarmParameters): object with crash relevant ego
            vehicle parameters.
        obstacle (HarmParameters): object with crash relevant obstacle
            parameters.
        pdof (float): Crash angle [rad].
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: Harm for ego vehicle
        float: Harm for obstacle vehicle
    """
    # calculate difference between pre-crash and post-crash speed
    ego_delta_v, obstacle_delta_v = calc_delta_v(
        vehicle_1=ego_vehicle, vehicle_2=obstacle, pdof=pdof
    )

    ego_harm = 1 / (
        1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * ego_delta_v)
    )

    obs_harm = 1 / (
        1
        + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * obstacle_delta_v)
    )

    return ego_harm, obs_harm


def get_unprotected_gidas_harm(ego_vehicle, obstacle, pdof, coeff):
    """
    MAIS 2+ model for pedestrians.

    Get the harm for two possible collision partners, if both the obstacle
    does not have a protective crash structure, via the logistic regression
    published by Bosch at the GIDAS symposia 2019.

    Args:
        ego_vehicle (HarmParameters): object with crash relevant ego
            vehicle parameters.
        obstacle (HarmParameters): object with crash relevant obstacle
            parameters.
        pdof (float): Crash angle [rad].
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: Harm for ego vehicle
        float: Harm for obstacle
    """
    # calculate difference between pre-crash and post-crash velocity
    ego_delta_v, obstacle_delta_v = calc_delta_v(
        vehicle_1=ego_vehicle, vehicle_2=obstacle, pdof=pdof
    )

    # calc ego harm
    ego_harm = 1 / (
        1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * ego_delta_v)
    )

    # calculate obstacle harm
    # logistic regression model
    obstacle_harm = 1 / (
        1
        + np.exp(
            coeff["pedestrian_MAIS2+"]["const"]
            - coeff["pedestrian_MAIS2+"]["speed"] * obstacle_delta_v
        )
    )

    return ego_harm, obstacle_harm
