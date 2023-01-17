"""Harm estimation function calling models based on risk json."""

import numpy as np
from commonroad.scenario.obstacle import ObstacleType

from EthicalTrajectoryPlanning.risk_assessment.helpers.harm_parameters import HarmParameters
from EthicalTrajectoryPlanning.risk_assessment.helpers.properties import calc_crash_angle, get_obstacle_mass
from EthicalTrajectoryPlanning.risk_assessment.utils.logistic_regression import (
    get_protected_log_reg_harm,
    get_unprotected_log_reg_harm,
)
from EthicalTrajectoryPlanning.risk_assessment.utils.reference_speed import (
    get_protected_ref_speed_harm,
    get_unprotected_ref_speed_harm,
)
from EthicalTrajectoryPlanning.risk_assessment.utils.reference_speed_symmetrical import (
    get_protected_inj_prob_ref_speed_complete_sym,
    get_protected_inj_prob_ref_speed_ignore_angle,
    get_protected_inj_prob_ref_speed_reduced_sym,
)
from EthicalTrajectoryPlanning.risk_assessment.utils.reference_speed_asymmetrical import (
    get_protected_inj_prob_ref_speed_complete,
    get_protected_inj_prob_ref_speed_reduced,
)
from EthicalTrajectoryPlanning.risk_assessment.utils.gidas import (
    get_protected_gidas_harm,
    get_unprotected_gidas_harm,
)
from EthicalTrajectoryPlanning.risk_assessment.utils.logistic_regression_symmetrical import (
    get_protected_inj_prob_log_reg_complete_sym,
    get_protected_inj_prob_log_reg_ignore_angle,
    get_protected_inj_prob_log_reg_reduced_sym,
)
from EthicalTrajectoryPlanning.risk_assessment.utils.logistic_regression_asymmetrical import (
    get_protected_inj_prob_log_reg_complete,
    get_protected_inj_prob_log_reg_reduced,
)


# Dictionary for existence of protective crash structure.
obstacle_protection = {
    ObstacleType.CAR: True,
    ObstacleType.TRUCK: True,
    ObstacleType.BUS: True,
    ObstacleType.BICYCLE: False,
    ObstacleType.PEDESTRIAN: False,
    ObstacleType.PRIORITY_VEHICLE: True,
    ObstacleType.PARKED_VEHICLE: True,
    ObstacleType.TRAIN: True,
    ObstacleType.MOTORCYCLE: False,
    ObstacleType.TAXI: True,
    ObstacleType.ROAD_BOUNDARY: None,
    ObstacleType.PILLAR: None,
    ObstacleType.CONSTRUCTION_ZONE: None,
    ObstacleType.BUILDING: None,
    ObstacleType.MEDIAN_STRIP: None,
    ObstacleType.UNKNOWN: False,
}


def harm_model(
    scenario,
    ego_vehicle_id: int,
    vehicle_params,
    ego_velocity: float,
    ego_yaw: float,
    obstacle_id: int,
    obstacle_size: float,
    obstacle_velocity: float,
    obstacle_yaw: float,
    pdof: float,
    ego_angle: float,
    obs_angle: float,
    modes,
    coeffs,
):
    """
    Get the harm for two possible collision partners.

    Args:
        scenario (Scenario): Considered scenario.
        ego_vehicle_id (Int): ID of ego vehicle.
        vehicle_params (Dict): Parameters of ego vehicle (1, 2 or 3).
        ego_velocity (Float): Velocity of ego vehicle [m/s].
        ego_yaw (Float): Yaw of ego vehicle [rad].
        obstacle_id (Int): ID of considered obstacle.
        obstacle_size (Float): Size of obstacle in [m²] (length * width)
        obstacle_velocity (Float): Velocity of obstacle [m/s].
        obstacle_yaw (Float): Yaw of obstacle [rad].
        pdof (float): Crash angle between ego vehicle and considered
            obstacle [rad].
        ego_angle (float): Angle of impact area for the ego vehicle.
        obs_angle (float): Angle of impact area for the obstacle.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json

    Returns:
        float: Harm for the ego vehicle.
        float: Harm for the other collision partner.
        HarmParameters: Class with independent variables for the ego
            vehicle
        HarmParameters: Class with independent variables for the obstacle
            vehicle
    """
    # create dictionaries with crash relevant parameters
    ego_vehicle = HarmParameters()
    obstacle = HarmParameters()

    # assign parameters to dictionary
    ego_vehicle.type = scenario.obstacle_by_id(ego_vehicle_id).obstacle_type
    obstacle.type = scenario.obstacle_by_id(obstacle_id).obstacle_type
    ego_vehicle.protection = obstacle_protection[ego_vehicle.type]
    obstacle.protection = obstacle_protection[obstacle.type]
    if ego_vehicle.protection is not None:
        ego_vehicle.mass = vehicle_params.m
        ego_vehicle.velocity = ego_velocity
        ego_vehicle.yaw = ego_yaw
        ego_vehicle.size = vehicle_params.w * vehicle_params.l
    else:
        ego_vehicle.mass = None
        ego_vehicle.velocity = None
        ego_vehicle.yaw = None
        ego_vehicle.size = None

    if obstacle.protection is not None:
        obstacle.velocity = obstacle_velocity
        obstacle.yaw = obstacle_yaw
        obstacle.size = obstacle_size
        obstacle.mass = get_obstacle_mass(
            obstacle_type=obstacle.type, size=obstacle.size
        )
    else:
        obstacle.mass = None
        obstacle.velocity = None
        obstacle.yaw = None
        obstacle.size = None

    # get model based on selection
    if modes["harm_mode"] == "log_reg":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_log_reg_harm(
                ego_vehicle=ego_vehicle,
                obstacle=obstacle,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_log_reg_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    elif modes["harm_mode"] == "ref_speed":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_ref_speed_harm(
                ego_vehicle=ego_vehicle,
                obstacle=obstacle,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_ref_speed_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    elif modes["harm_mode"] == "gidas":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_gidas_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_gidas_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    else:
        raise ValueError(
            "Please select a valid mode for harm estimation "
            "(log_reg, ref_speed, gidas)"
        )

    return ego_vehicle.harm, obstacle.harm, ego_vehicle, obstacle


def get_harm(scenario, traj, predictions, ego_id, vehicle_params, modes, coeffs, timer):
    """Get harm.

    Args:
        scenario (_type_): _description_
        traj (_type_): _description_
        predictions (_type_): _description_
        ego_id (_type_): _description_
        vehicle_params (_type_): _description_
        modes (_type_): _description_
        coeffs (_type_): _description_
        timer (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get the IDs of the predicted obstacles
    obstacle_ids = list(predictions.keys())

    # max_pred_length = 0

    ego_harm_traj = {}
    obst_harm_traj = {}

    # get the ego vehicle size
    # ego_vehicle_size = vehicle_params.w * vehicle_params.l

    for obstacle_id in obstacle_ids:
        # choose which model should be used to calculate the harm
        ego_harm_fun, obstacle_harm_fun = get_model(modes, obstacle_id, scenario)
        # only calculate the risk as long as both obstacles are in the scenario
        pred_path = predictions[obstacle_id]['pos_list']
        pred_length = min(len(traj.t) - 1, len(pred_path))
        if pred_length == 0:
            continue

        # get max prediction length
        # if pred_length > max_pred_length:
        #     max_pred_length = pred_length

        # get the size, the velocity and the orientation of the predicted
        # vehicle
        pred_size = (
            predictions[obstacle_id]['shape']['length']
            * predictions[obstacle_id]['shape']['width']
        )
        pred_v = np.array(predictions[obstacle_id]['v_list'], dtype=np.float)
        pred_yaw = np.array(predictions[obstacle_id]['orientation_list'], dtype=np.float)

        # lists to save ego and obstacle harm as well as ego and obstacle risk
        # one list per obstacle
        ego_harm_obst = []
        obst_harm_obst = []

        # replace get_obstacle_mass() by get_obstacle_mass()
        # get the predicted obstacle vehicle mass
        obstacle_mass = get_obstacle_mass(
            obstacle_type=scenario.obstacle_by_id(obstacle_id).obstacle_type, size=pred_size
        )

        # calc crash angle if comprehensive mode selected
        if modes["crash_angle_simplified"] is False:

            with timer.time_with_cm(
                "simulation/sort trajectories/calculate costs/calculate risk/"
                + "calculate harm/calculate PDOF comp"
            ):

                pdof, ego_angle, obs_angle = calc_crash_angle(
                    traj=traj,
                    predictions=predictions,
                    scenario=scenario,
                    obstacle_id=obstacle_id,
                    modes=modes,
                    vehicle_params=vehicle_params,
                )

            for i in range(pred_length):
                with timer.time_with_cm(
                    "simulation/sort trajectories/calculate costs/calculate risk/"
                    + "calculate harm/harm_model"
                ):

                    # get the harm ego harm and the harm of the collision opponent
                    ego_harm, obst_harm, ego_harm_data, obst_harm_data = harm_model(
                        scenario=scenario,
                        ego_vehicle_id=ego_id,
                        vehicle_params=vehicle_params,
                        ego_velocity=traj.v[i],
                        ego_yaw=traj.yaw[i],
                        obstacle_id=obstacle_id,
                        obstacle_size=pred_size,
                        obstacle_velocity=pred_v[i],
                        obstacle_yaw=pred_yaw[i],
                        pdof=pdof,
                        ego_angle=ego_angle,
                        obs_angle=obs_angle,
                        modes=modes,
                        coeffs=coeffs,
                    )

                    # store information to calculate harm and harm value in list
                    ego_harm_obst.append(ego_harm)
                    obst_harm_obst.append(obst_harm)
        else:
            # calc the risk for every time step
            with timer.time_with_cm(
                    "simulation/sort trajectories/calculate costs/calculate risk/"
                    + "calculate harm/calculate PDOF simple"
            ):
                # crash angle between ego vehicle and considered obstacle [rad]
                pdof_array = predictions[obstacle_id]["orientation_list"][:pred_length] - traj.yaw[:pred_length] + np.pi
                rel_angle_array = np.arctan2(predictions[obstacle_id]["pos_list"][:pred_length, 1] - traj.y[:pred_length],
                                              predictions[obstacle_id]["pos_list"][:pred_length, 0] - traj.x[:pred_length])
                # angle of impact area for the ego vehicle
                ego_angle_array = rel_angle_array - traj.yaw[:pred_length]
                # angle of impact area for the obstacle
                obs_angle_array = np.pi + rel_angle_array - predictions[obstacle_id]["orientation_list"][:pred_length]

                # calculate the difference between pre-crash and post-crash speed
                delta_v_array = np.sqrt(
                    np.power(traj.v[:pred_length], 2)
                    + np.power(pred_v[:pred_length], 2)
                    + 2 * traj.v[:pred_length] * pred_v[:pred_length] * np.cos(pdof_array)
                )
                ego_delta_v = obstacle_mass / (vehicle_params.m + obstacle_mass) * delta_v_array
                obstacle_delta_v = vehicle_params.m / (vehicle_params.m + obstacle_mass) * delta_v_array

                # calculate harm besed on selected model
                ego_harm_obst = ego_harm_fun(velocity=ego_delta_v, angle=ego_angle_array, coeff=coeffs)
                obst_harm_obst = obstacle_harm_fun(velocity=obstacle_delta_v, angle=obs_angle_array, coeff=coeffs)
        # store harm list for the obstacles in dictionary for current frenét
        # trajectory
        ego_harm_traj[obstacle_id] = ego_harm_obst
        obst_harm_traj[obstacle_id] = obst_harm_obst

    return ego_harm_traj, obst_harm_traj


def get_model(modes, obstacle_id, scenario):
    """Get harm model according to settings.

    Args:
        modes (_type_): _description_
        obstacle_id (_type_): _description_
        scenario (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # obstacle protection type
    obs_protection = obstacle_protection[scenario.obstacle_by_id(obstacle_id).obstacle_type]

    if modes["harm_mode"] == "log_reg":
        # select case based on protection structure
        if obs_protection is True:
            # calculate harm based on angle mode
            if modes["ignore_angle"] is False:
                if modes["sym_angle"] is False:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_complete

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_complete

                    else:
                        # use log reg reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_reduced

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_reduced
                else:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg sym complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_complete_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_complete_sym
                    else:
                        # use log reg sym reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_reduced_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_reduced_sym
            else:
                # use log reg delta v
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_log_reg_ignore_angle

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_log_reg_ignore_angle

        elif obs_protection is False:
            # calc ego harm
            ego_harm = get_protected_inj_prob_log_reg_ignore_angle

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian"]["const"]
                    - coeff["pedestrian"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity,angle,coeff : 1  # noqa E731
            obstacle_harm = lambda velocity,angle,coeff : 1  # noqa E731

    elif modes["harm_mode"] == "ref_speed":
        # select case based on protection structure
        if obs_protection is True:
            # calculate harm based on angle mode
            if modes["ignore_angle"] is False:
                if modes["sym_angle"] is False:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_complete

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_complete

                    else:
                        # use log reg reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_reduced

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_reduced
                else:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg sym complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_complete_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_complete_sym
                    else:
                        # use log reg sym reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_reduced_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_reduced_sym
            else:
                # use log reg delta v
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_ref_speed_ignore_angle

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_ref_speed_ignore_angle

        elif obs_protection is False:
            # calc ego harm
            ego_harm = get_protected_inj_prob_ref_speed_ignore_angle

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian"]["const"]
                    - coeff["pedestrian"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity,angle,coeff : 1  # noqa E731
            obstacle_harm = lambda velocity,angle,coeff : 1  # noqa E731

    elif modes["harm_mode"] == "gidas":
        # select case based on protection structure
        if obs_protection is True:
            ego_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )

            obs_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1
                + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )
        elif obs_protection is False:
            # calc ego harm
            ego_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian_MAIS2+"]["const"]
                    - coeff["pedestrian_MAIS2+"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity, angle, coeff: 1  # noqa E731
            obstacle_harm = lambda velocity, angle, coeff: 1  # noqa E731

    else:
        raise ValueError(
            "Please select a valid mode for harm estimation "
            "(log_reg, ref_speed, gidas)"
        )

    return ego_harm, obstacle_harm
