"""Function to create figures in "Bilder/results"."""

import matplotlib.pyplot as plt
import numpy as np
import os

# global dict for risks
risks = {}
risk_cost_dict = {
    "time_list": [],
    "bayes_list": [],
    "equality_list": [],
    "maximin_list": [],
    "responsibility_list": [],
    "ego_list": [],
}


def risk_dashboard(
    scenario,
    time_step: int,
    destination: str,
    risk_modes,
    weights,
    planning_problem=None,
    traj=None,
):
    """
    Create a dashboard showing risk for traffic participants.

    Create a dashboard to visualize the risk for all traffic participants.
    Saves .png files, which can be converted to a GIF via GIF-maker. Creates a
    subfolder named according to the scenario ID in the destination folder.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        planning_problem (PlanningProblem): Considered planning problem.
            Defaults to None.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.

    Returns:
            No return value.
    """
    if risk_modes["risk_dashboard"] is True:

        if len(traj) > 0:

            global risks
            global risk_cost_dict

            # clear everything
            plt.cla()

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

            # Calculate ego risk value from risk coarse
            ego_risk_dict = traj[0].ego_risk_dict

            if "Ego" not in risks:
                risks["Ego"] = [[0], [0]]
                risks["Ego"][0][0] = time_step * scenario.dt
                risks["Ego"][1][0] = sum(ego_risk_dict.values())

            else:
                risks["Ego"][0].append(time_step * scenario.dt)
                risks["Ego"][1].append(sum(ego_risk_dict.values()))

            # Calculate obstacle risk values
            risk_dict = traj[0].obst_risk_dict

            # iterate through data dictionary to extract risk for obstacles
            for obstacle_id, risk_value in risk_dict.items():

                if obstacle_id not in risks:
                    risks[obstacle_id] = [[0], [0]]
                    risks[obstacle_id][0][0] = time_step * scenario.dt
                    risks[obstacle_id][1][0] = risk_value

                else:
                    risks[obstacle_id][0].append(time_step * scenario.dt)
                    risks[obstacle_id][1].append(risk_value)

            for key in risks.keys():
                if key == "Ego":
                    # plot risk for ego vehicle
                    ax1.plot(risks[key][0], risks[key][1], label=key)
                else:
                    # plot risk for each obstacle
                    ax1.plot(risks[key][0], risks[key][1], label="Obstacle " + str(key))

            # get the target time to show it in the title
            if hasattr(planning_problem.goal.state_list[0], 'time_step'):
                target_time = (
                    planning_problem.goal.state_list[0].time_step.end * scenario.dt
                )
                ax1.set_xlim((0, target_time * 1.1))
                ax2.set_xlim((0, target_time * 1.1))

            ax1.legend()
            ax1.set_ylabel("Risk ({})".format(risk_modes['trajectory_risk']))
            ax1.set_xlabel("Time in s")
            ax1.set_ylim((0, risk_modes['max_acceptable_risk'] * 1.2))
            ax1.plot(
                [0, target_time * 1.1], [risk_modes['max_acceptable_risk']] * 2, "r--"
            )

            # ---- 2nd axis for risk costs ----

            if risk_modes['trajectory_risk'] == 'mean':
                risk_operator = np.mean
            elif risk_modes['trajectory_risk'] == 'max':
                risk_operator = np.max
                risk_operator_resp = np.min
            else:
                raise NotImplementedError

            risk_cost_dict["time_list"].append(time_step * scenario.dt)
            risk_cost_dict["bayes_list"].append(
                risk_operator(traj[0].cost_dict["risk_cost_dict"]['bayes'])
                * weights["bayes"]
            )
            risk_cost_dict["equality_list"].append(
                risk_operator(traj[0].cost_dict["risk_cost_dict"]['equality'])
                * weights["equality"]
            )
            risk_cost_dict["maximin_list"].append(
                risk_operator(traj[0].cost_dict["risk_cost_dict"]['maximin'])
                * weights["maximin"]
            )
            risk_cost_dict["responsibility_list"].append(
                risk_operator_resp(
                    traj[0].cost_dict["risk_cost_dict"]['responsibility']
                )
                * weights["responsibility"]
                * weights["bayes"]
            )
            risk_cost_dict["ego_list"].append(
                risk_operator(traj[0].cost_dict["risk_cost_dict"]['ego'])
                * weights["ego"]
            )

            ax2.plot(
                risk_cost_dict["time_list"],
                risk_cost_dict["bayes_list"],
                label="Bayes Risk (weight = {})".format(weights["bayes"]),
            )
            ax2.plot(
                risk_cost_dict["time_list"],
                risk_cost_dict["equality_list"],
                label="Equality Risk (weight = {})".format(weights["equality"]),
            )
            ax2.plot(
                risk_cost_dict["time_list"],
                risk_cost_dict["maximin_list"],
                label="Maximin Risk (weight = {})".format(weights["maximin"]),
            )
            ax2.plot(
                risk_cost_dict["time_list"],
                risk_cost_dict["responsibility_list"],
                label="Responsibility Risk (weight = {})".format(
                    weights["responsibility"]
                ),
            )
            ax2.plot(
                risk_cost_dict["time_list"],
                risk_cost_dict["ego_list"],
                label="Ego Risk (weight = {})".format(weights["ego"]),
            )

            ax2.legend()
            ax2.set_xlabel("Time in s")
            ax2.set_ylabel("Risk Costs")

            # Create directory for pictures
            destination = os.path.join(destination, str(scenario.benchmark_id))
            if not os.path.exists(destination):
                os.makedirs(destination)

            picture_path = destination + "/Risk_Dashboard_" + str(time_step)
            plt.savefig(picture_path)
            plt.close()
