"""Calculate the risk for an according object."""
import numpy as np


def calc_obstacle_risk(
    risk_dict, ego=False, trajectory_risk_mode="max", scale_factor=1.0
):
    """Calculate the risk for any obstacle.

    Args:
        risk_dict ([dict]): [Stores different obstacles as keys and their temporal risk coarse]
        ego (bool, optional): [Sums up the risk if set to true]. Defaults to False.
        trajectory_risk_mode (str, optional): [Take mean or maximum risk of risk coarse]. Defaults to "max".
        scale_factor (float, optional): [Scale down future risk by scalefactor(<=1)**time_step]. Defaults to 1.0.

    Raises:
        NotImplementedError: [description]

    Returns:
        [dict]: [obstacles as key and one risk as value]
    """
    if trajectory_risk_mode == 'mean':
        risk_operator = np.mean
    elif trajectory_risk_mode == 'max':
        risk_operator = np.max
    else:
        raise NotImplementedError

    risk = 0
    return_dict = {}

    for obstalce_id, risk_over_time in risk_dict.items():
        risk_list = [
            ts.risk * np.power(scale_factor, c)
            for c, ts in enumerate(risk_over_time)
            if ts is not None
        ]

        # For ego we have to sum up the risks for all possible collisions
        if ego:
            risk += risk_operator(risk_list)
        else:
            return_dict[obstalce_id] = risk_operator(risk_list)

    if ego:
        return_dict['ego'] = risk

    return return_dict
