"""Harm estimation function calling models based on risk json."""

from commonroad.scenario.obstacle import ObstacleType

from EthicalTrajectoryPlanning.risk_assessment.helpers.harm_parameters import HarmParameters
from EthicalTrajectoryPlanning.risk_assessment.helpers.properties import get_obstacle_mass
from EthicalTrajectoryPlanning.risk_assessment.utils.logistic_regression import (
    get_protected_log_reg_harm,
    get_unprotected_log_reg_harm,
)
from EthicalTrajectoryPlanning.risk_assessment.utils.reference_speed import (
    get_protected_ref_speed_harm,
    get_unprotected_ref_speed_harm,
)
from EthicalTrajectoryPlanning.risk_assessment.utils.gidas import (
    get_protected_gidas_harm,
    get_unprotected_gidas_harm,
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
