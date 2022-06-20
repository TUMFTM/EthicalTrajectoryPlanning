"""Risk cost function and principles of ethics of risk."""

from EthicalTrajectoryPlanning.planner.Frenet.utils.validity_checks import create_collision_object
from commonroad_dc.collision.trajectory_queries import trajectory_queries

from EthicalTrajectoryPlanning.risk_assessment.harm_estimation import harm_model
from EthicalTrajectoryPlanning.risk_assessment.collision_probability import (
    get_collision_probability,
)
from EthicalTrajectoryPlanning.risk_assessment.helpers.properties import (
    calc_crash_angle,
    calc_crash_angle_simple,
)
from EthicalTrajectoryPlanning.risk_assessment.helpers.timers import ExecTimer
from EthicalTrajectoryPlanning.planner.utils.responsibility import assign_responsibility_by_action_space, calc_responsibility_reach_set
from EthicalTrajectoryPlanning.risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle


def calc_risk(
    traj,
    ego_state,
    predictions: dict,
    scenario,
    ego_id: int,
    vehicle_params,
    params,
    road_boundary,
    reach_set=None,
    exec_timer=None,
):
    """
    Calculate the risk for the given trajectory.

    Args:
        traj (FrenetTrajectory): Considered frenét trajectory.
        predictions (dict): Predictions for the visible obstacles.
        scenario (Scenario): Considered scenario.
        ego_id (int): ID of the ego vehicle.
        vehicle_params (VehicleParameters): Vehicle parameters of the
            considered vehicle.
        weights (Dict): Weighing factors. Read from weights.json.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json.
        exec_timer (ExecTimer): Timer for the exec_timing.json.

    Returns:
        float: Weighed risk costs.
        dict: Dictionary with ego harms for every time step concerning every
            obstacle
        dict: Dictionary with obstacle harms for every time step concerning
            every obstacle

    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    modes = params['modes']
    coeffs = params['harm']

    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate risk/"
        + "collision probability"
    ):
        coll_prob_dict = get_collision_probability(
            traj=traj,
            predictions=predictions,
            vehicle_params=vehicle_params,
        )

    ego_harm_traj, obst_harm_traj = get_harm(
        scenario, traj, predictions, ego_id, vehicle_params, modes, coeffs, timer
    )

    # Calculate risk out of harm and collision probability
    ego_risk_traj = {}
    obst_risk_traj = {}
    ego_risk_max = {}
    obst_risk_max = {}
    ego_harm_max = {}
    obst_harm_max = {}

    for key in ego_harm_traj:
        ego_risk_traj[key] = [
            ego_harm_traj[key][t] * coll_prob_dict[key][t]
            for t in range(len(ego_harm_traj[key]))
        ]
        obst_risk_traj[key] = [
            obst_harm_traj[key][t] * coll_prob_dict[key][t]
            for t in range(len(obst_harm_traj[key]))
        ]

        # Take max as representative for the whole trajectory
        ego_risk_max[key] = max(ego_risk_traj[key])
        obst_risk_max[key] = max(obst_risk_traj[key])
        ego_harm_max[key] = max(ego_harm_traj[key])
        obst_harm_max[key] = max(obst_harm_traj[key])

    # calculate boundary harm
    col_obj = create_collision_object(traj, vehicle_params, ego_state)

    leaving_road_at = trajectory_queries.trajectories_collision_static_obstacles(
        trajectories=[col_obj],
        static_obstacles=road_boundary,
        method="grid",
        num_cells=32,
        auto_orientation=True,
    )

    if leaving_road_at[0] != -1:
        coll_time_step = leaving_road_at[0] - ego_state.time_step
        coll_vel = traj.v[coll_time_step]

        boundary_harm = get_protected_inj_prob_log_reg_ignore_angle(
            velocity=coll_vel, coeff=coeffs
        )

    else:
        boundary_harm = 0

    return ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, boundary_harm


def get_bayesian_costs(ego_risk_max, obst_risk_max, boundary_harm):
    """
    Bayesian Principle.

    Calculate the risk cost via the Bayesian Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.

    Returns:
        Dict: Risk costs for the considered trajectory according to the
            Bayesian Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return (sum(ego_risk_max.values()) + sum(obst_risk_max.values()) + boundary_harm) / (
        len(ego_risk_max) * 2
    )


def get_equality_costs(ego_risk_max, obst_risk_max):
    """
    Equality Principle.

    Calculate the risk cost via the Equality Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        equality_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Equality Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(
        [abs(ego_risk_max[key] - obst_risk_max[key]) for key in ego_risk_max]
    ) / len(ego_risk_max)


def get_maximin_costs(ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, boundary_harm, eps=10e-10, scale_factor=10):
    """
    Maximin Principle.

    Calculate the risk cost via the Maximin principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        maximin_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Maximum Principle
    """
    if len(ego_harm_max) == 0:
        return 0

    # Set maximin to 0 if probability (or risk) is 0
    maximin_ego = [a * int(b < eps) for a, b in zip(ego_harm_max.values(), ego_risk_max.values())]
    maximin_obst = [a * int(bool(b < eps)) for a, b in zip(obst_harm_max.values(), obst_risk_max.values())]

    return max(maximin_ego + maximin_obst + [boundary_harm])**scale_factor


def get_ego_costs(ego_risk_max, boundary_harm):
    """
    Calculate the utilitarian ego cost for the given trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.

    Returns:
        Dict: Utilitarian ego risk cost
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(ego_risk_max.values()) + boundary_harm


def get_responsibility_cost(scenario, traj, ego_state, obst_risk_max, predictions, reach_set, mode="reach_set"):
    """Get responsibility cost.

    Args:
        obst_risk_max (_type_): _description_
        predictions (_type_): _description_
        mode (str) : "reach set" for reachable set mode, else assignement by space of action

    Returns:
        _type_: _description_

    """
    if "reach_set" in mode and reach_set is not None:
        resp_cost = calc_responsibility_reach_set(traj, ego_state, reach_set)

    else:
        # Assign responsibility to predictions
        predictions = assign_responsibility_by_action_space(
            scenario, ego_state, predictions
        )
        resp_cost = 0

        for key in predictions:
            resp_cost -= predictions[key]["responsibility"] * obst_risk_max[key]

    return resp_cost


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

    max_pred_length = 0

    ego_harm_traj = {}
    obst_harm_traj = {}
    for obstacle_id in obstacle_ids:

        # only calculate the risk as long as both obstacles are in the scenario
        pred_path = predictions[obstacle_id]['pos_list']
        pred_length = min(len(traj.t) - 1, len(pred_path))
        if pred_length == 0:
            continue

        # get max prediction length
        if pred_length > max_pred_length:
            max_pred_length = pred_length

        # get the size, the velocity and the orientation of the predicted
        # vehicle
        pred_size = (
            predictions[obstacle_id]['shape']['length']
            * predictions[obstacle_id]['shape']['width']
        )
        pred_v = predictions[obstacle_id]['v_list']
        pred_yaw = predictions[obstacle_id]['orientation_list']

        # lists to save ego and obstacle harm as well as ego and obstacle risk
        # one list per obstacle
        ego_harm_obst = []
        obst_harm_obst = []

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

        # calc the risk for every time step
        for i in range(pred_length):

            # calc crash angle simplified if selected
            if modes["crash_angle_simplified"] is True:

                with timer.time_with_cm(
                    "simulation/sort trajectories/calculate costs/calculate risk/"
                    + "calculate harm/calculate PDOF simple"
                ):

                    pdof, ego_angle, obs_angle = calc_crash_angle_simple(
                        traj=traj,
                        predictions=predictions,
                        obstacle_id=obstacle_id,
                        time_step=i,
                    )

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

        # store harm list for the obstacles in dictionary for current frenét
        # trajectory
        ego_harm_traj[obstacle_id] = ego_harm_obst
        obst_harm_traj[obstacle_id] = obst_harm_obst

    return ego_harm_traj, obst_harm_traj
