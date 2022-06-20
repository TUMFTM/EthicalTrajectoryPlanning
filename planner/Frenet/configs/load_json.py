"""Helper functions to load jsons."""

import json
import numpy as np
import os


def load_risk_json():
    """
    Load the risk.json with harm weights.

    Returns:
        Dict: weights and modes form risk.json
    """
    risk_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "risk.json",
    )
    with open(risk_config_path, "r") as f:
        jsondata = json.load(f)

    return jsondata


def load_harm_parameter_json():
    """
    Load the harm_parameters.json with model parameters.

    Returns:
        Dict: model parameters from parameter.json
    """
    parameter_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "harm_parameters.json",
    )
    with open(parameter_config_path, "r") as f:
        jsondata = json.load(f)
    return jsondata


def load_weight_json(filename="weights.json"):
    """
    Load the weights.json with cost weights for risk.

    Returns:
        Dict: model parameters from weights.json
    """
    weight_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename,
    )
    with open(weight_config_path, "r") as f:
        jsondata = json.load(f)

    print(f"\nLoaded weights from {weight_config_path}")

    return jsondata


def load_planning_json(filename="planning.json"):
    """
    Load the planning.json with modes.

    Returns:
        Dict: parameters from planning.json
    """
    planning_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename,
    )
    with open(planning_config_path, "r") as f:
        jsondata = json.load(f)

    # Create the d_list with linspace
    jsondata["frenet_settings"]["frenet_parameters"]["d_list"] = np.linspace(
        -jsondata["frenet_settings"]["frenet_parameters"]["d_list"]["d_max_abs"],
        jsondata["frenet_settings"]["frenet_parameters"]["d_list"]["d_max_abs"],
        jsondata["frenet_settings"]["frenet_parameters"]["d_list"]["n"],
    )

    return jsondata


# EOF
