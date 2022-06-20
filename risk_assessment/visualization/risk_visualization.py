
"""Function to create figures in "Bilder/results"."""

import os
import matplotlib.pyplot as plt
import numpy as np
from commonroad.visualization.draw_dispatch_cr import draw_object

# color matching for frenét trajectories
col = ['green', 'greenyellow', 'yellow', 'orange', 'red']


def create_risk_files(scenario,
                      time_step: int,
                      destination: str,
                      risk_modes,
                      weights,
                      marked_vehicle: [int] = None,
                      planning_problem=None,
                      traj=None,
                      fut_pos_list: np.ndarray = None,
                      visible_area=None,
                      global_path: np.ndarray = None,
                      global_path_after_goal: np.ndarray = None,
                      driven_traj=None):
    """
    Create plots to visualize the choosen Frenét traj and its risks.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        weights (Dict): Read from weights.json.
        marked_vehicle ([int]): IDs of the marked vehicles.
            Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem.
            Defaults to None.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.
        fut_pos_list (np.ndarray): Future positions of the vehicles.
            Defaults to None.
        visible_area (shapely.Polygon): Polygon of the visible area.
            Defaults to None.
        global_path (np.ndarray): Global path for the planning problem.
            Defaults to None.
        global_path_after_goal (np.ndarray): Global path for the planning
            problem after reaching the goal.
            Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego vehicle.
            Defaults to None.

    Returns:
        No return value.
    """
    if risk_modes["figures"]["create_figures"] is True:

        create_scenario_figure(scenario=scenario,
                               time_step=time_step,
                               destination=destination,
                               risk_modes=risk_modes,
                               marked_vehicle=marked_vehicle,
                               planning_problem=planning_problem,
                               traj=traj,
                               fut_pos_list=fut_pos_list,
                               visible_area=visible_area,
                               global_path=global_path,
                               global_path_after_goal=global_path_after_goal,
                               driven_traj=driven_traj)

        create_partial_chart(scenario=scenario,
                             time_step=time_step,
                             destination=destination,
                             risk_modes=risk_modes,
                             traj=traj)

        create_cost_chart(scenario=scenario,
                          time_step=time_step,
                          destination=destination,
                          weights=weights,
                          traj=traj)

        create_total_cost_chart(scenario=scenario,
                                time_step=time_step,
                                destination=destination,
                                risk_modes=risk_modes,
                                traj=traj)


def create_scenario_figure(scenario,
                           time_step: int,
                           destination: str,
                           risk_modes,
                           marked_vehicle: [int] = None,
                           planning_problem=None,
                           traj=None,
                           fut_pos_list: np.ndarray = None,
                           visible_area=None,
                           global_path: np.ndarray = None,
                           global_path_after_goal: np.ndarray = None,
                           driven_traj=None):
    """
    Create a figure with the most-costefficient Frenét trajectories.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        marked_vehicle ([int]): IDs of the marked vehicles. Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem.
            Defaults to None.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.
        fut_pos_list (np.ndarray): Future positions of the vehicles.
            Defaults to None.
        visible_area (shapely.Polygon): Polygon of the visible area.
            Defaults to None.
        global_path (np.ndarray): Global path for the planning problem.
            Defaults to None.
        global_path_after_goal (np.ndarray): Global path for the planning
            problem after reaching the goal. Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego vehicle.
            Defaults to None.

    Returns:
        No return value.
    """
    if traj is not None:
        # check if enough trajectories are available to plot
        if risk_modes["figures"]["number_plotted_trajectories"] > len(traj):
            number = len(traj)
        else:
            number = risk_modes["figures"]["number_plotted_trajectories"]
    else:
        number = 0

    # clear everything
    plt.cla()

    # set plot limits to show the road section around the ego vehicle
    plot_limits = [driven_traj[-1].position[0] - 20,
                   driven_traj[-1].position[0] + 20,
                   driven_traj[-1].position[1] - 20,
                   driven_traj[-1].position[1] + 20]

    # plot the scenario at the current time step
    draw_object(scenario,
                draw_params={'time_begin': time_step, 'scenario':
                             {'dynamic_obstacle': {'show_label': False}}},
                plot_limits=plot_limits)
    plt.gca().set_aspect('equal')

    # draw the planning problem
    if planning_problem is not None:
        draw_object(planning_problem)

    # mark the ego vehicle
    if marked_vehicle is not None:
        draw_object(obj=scenario.obstacle_by_id(marked_vehicle),
                    draw_params={'time_begin': time_step,
                                 'facecolor': 'g'})

    # Draw global path
    if global_path is not None:
        plt.plot(global_path[:, 0], global_path[:, 1], color='blue',
                 zorder=20, label='global path')
        if global_path_after_goal is not None:
            plt.plot(global_path_after_goal[:, 0],
                     global_path_after_goal[:, 1], color='blue', zorder=20,
                     linestyle='--')

    # draw driven trajectory
    if driven_traj is not None:
        x = [state.position[0] for state in driven_traj]
        y = [state.position[1] for state in driven_traj]
        plt.plot(x, y, color='green', zorder=25)

    # draw planned trajectory
    if traj is not None:
        for i in range(number):
            if i == 0:
                plt.plot(traj[i].x, traj[i].y, alpha=1., color=col[i],
                         zorder=25 - i, lw=3.,
                         label='Chosen trajectory')
            else:
                plt.plot(traj[i].x, traj[i].y, alpha=1., color=col[i],
                         zorder=25 - i, lw=3.,
                         label='Trajectory ' + str(i + 1))

    # draw predictions
    if fut_pos_list is not None:
        for fut_pos in fut_pos_list:
            plt.plot(fut_pos[:, 0], fut_pos[:, 1], '.c', markersize=2,
                     alpha=0.8)

    # draw visible sensor area
    if visible_area is not None:
        if visible_area.geom_type == 'MultiPolygon':
            for geom in visible_area.geoms:
                plt.fill(*geom.exterior.xy, 'g', alpha=0.2, zorder=10)
        elif visible_area.geom_type == 'Polygon':
            plt.fill(*visible_area.exterior.xy, 'g', alpha=0.2, zorder=10)
        else:
            for obj in visible_area:
                if obj.geom_type == 'Polygon':
                    plt.fill(*obj.exterior.xy, 'g', alpha=0.2, zorder=10)

    # get the target time to show it in the title
    if hasattr(planning_problem.goal.state_list[0], 'time_step'):
        target_time_string = ('Target-time: %.1f s - %.1f s' %
                              (planning_problem.goal.state_list[0].
                               time_step.start * scenario.dt,
                               planning_problem.goal.state_list[0].
                               time_step.end * scenario.dt))
    else:
        target_time_string = 'No target-time'

    plt.legend()
    plt.title('Time: {0:.1f} s'.format(time_step * scenario.dt) + '    ' +
              target_time_string)

    # Create directory for pictures
    destination = os.path.join(destination, str(scenario.benchmark_id))
    if not os.path.exists(destination):
        os.makedirs(destination)

    picture_path = destination + "/Figure_" + \
        str(time_step)
    if not os.path.exists(picture_path + ".png"):
        plt.savefig(picture_path)
    else:
        for i in range(1, 10):
            if not os.path.exists(picture_path + "-" + str(i) + ".png"):
                plt.savefig(picture_path + "-" + str(i) + ".png")
                break
    plt.close()


def create_partial_chart(scenario,
                         time_step: int,
                         destination: str,
                         risk_modes,
                         traj=None):
    """
    Create a chart with partial harm, collision probability, and risks.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.

    Returns:
        No return value.
    """
    if traj is not None:
        # check if enough trajectories are available to plot
        if risk_modes["figures"]["number_plotted_trajectories"] > len(traj):
            number = len(traj)
        else:
            number = risk_modes["figures"]["number_plotted_trajectories"]
    else:
        number = 0

    # only add chart if a valid Frenét trajectory exists
    if number > 0:

        # add subplots
        fig, ((ax1, ax3, ax5, ax7), (ax2, ax4, ax6, ax8)) = \
            plt.subplots(nrows=2, ncols=4)
        fig.set_size_inches(11.69 * 2, 8.27 * 2)

        # get plots for the ego vehicle
        # variables to be plotted are data_harm, ego_vehicle_data,
        # and data_risk
        ego_vehicle_data = ""

        # iterate through data dictionary to extract harm and risk,
        # and assign it to plot variables
        for obstacle_id, harm_dict in traj[0].ego_harm_dict.items():
            data_harm = []
            data_risk = []
            data_prob = []
            for ts in harm_dict:
                if ts is not None:
                    data_harm.append(ts.harm)
                    data_risk.append(ts.risk)
                    data_prob.append(ts.prob)
                else:
                    data_harm.append(0)
                    data_risk.append(0)
                    data_prob.append(0)

            # plot ego harm and ego risk for each obstacle
            ax1.plot(data_harm, label="Obstacle " + obstacle_id)
            ax3.plot(data_prob, label="Obstacle " + obstacle_id)
            ax5.plot(data_risk, label="Obstacle " + obstacle_id)

            # create string with ego data for harm evaluation of next
            # time step
            if ego_vehicle_data == "":
                ego_vehicle_data += "Mass: " + str(harm_dict[0].mass) + \
                    "\nVelocity: " + str(harm_dict[0].velocity) + \
                    "\nYaw: " + str(harm_dict[0].yaw) + \
                    "\nSize: " + str(harm_dict[0].size) + \
                    "\nHarm: " + str(harm_dict[0].harm) + \
                    "\nRisk: " + str(harm_dict[0].risk) + "\n\n"
            else:
                ego_vehicle_data += "Harm: " + str(harm_dict[0].harm) + \
                                    "\nRisk: " + str(harm_dict[0].risk) + \
                                    "\n\n"

        # add description of ego harm plot
        ax1.legend(loc='upper right')
        ax1.set_ylabel("ego harm for different obstacles")

        # add description of ego prob plot
        ax3.legend(loc='upper right')
        ax3.set_ylabel("collision probability for different obstacles")

        # add description of ego risk plot
        ax5.legend(loc='upper right')
        ax5.set_ylabel("ego risk for different obstacles")

        # get plots for the obstacles
        # variables to be plotted are data_harm, obst_vehicle_data, and
        # data_risk
        obst_vehicle_data = ""

        # iterate through data dictionary to extract harm and risk,
        # and assign it to plot variables
        for obstacle_id, harm_dict in traj[0].obst_harm_dict.items():
            data_harm = []
            data_risk = []
            data_prob = []
            for ts in harm_dict:
                if ts is not None:
                    data_harm.append(ts.harm)
                    data_risk.append(ts.risk)
                    data_prob.append(ts.prob)
                else:
                    data_harm.append(0)
                    data_risk.append(0)
                    data_prob.append(0)

            # plot harm and risk for each obstacle for collisions with
            # the ego vehicle
            ax2.plot(data_harm, label="Obstacle " + obstacle_id)
            ax4.plot(data_prob, label="Obstacle " + obstacle_id)
            ax6.plot(data_risk, label="Obstacle " + obstacle_id)

            # create string with obstacle data for harm evaluation of next
            # time step
            obst_vehicle_data += str(harm_dict[0].type) + ", Mass: " + \
                str(harm_dict[0].mass) + "\nVelocity: " + \
                str(harm_dict[0].velocity) + "\nYaw: " + \
                str(harm_dict[0].yaw) + "\nSize: " + \
                str(harm_dict[0].size) + "\nHarm: " + \
                str(harm_dict[0].harm) + "\nRisk: " + \
                str(harm_dict[0].risk) + "\n\n"

        # add description of obstacle harm plot
        ax2.legend(loc='upper right')
        ax2.set_ylabel("obstacle harm")

        # add description of ego prob plot
        ax4.legend(loc='upper right')
        ax4.set_ylabel("collision probability for different obstacles")

        # add description of obstacle risk plot
        ax6.legend(loc='upper right')
        ax6.set_ylabel("obstacle risk")

        # add ego vehicle and obstacle data in subplots 3 and 4
        ax7.axis('off')
        ax7.text(0, 1, ego_vehicle_data, verticalalignment='top',
                 fontsize=8)
        ax8.axis('off')
        ax8.text(0, 1, obst_vehicle_data, verticalalignment='top',
                 fontsize=8)

        fig.suptitle("Harm for ego vehicle and obstacles")

        # Create directory for pictures
        destination = os.path.join(destination, str(scenario.benchmark_id))
        if not os.path.exists(destination):
            os.makedirs(destination)

        picture_path = destination + "/Partial_" + str(time_step)
        if not os.path.exists(picture_path + ".png"):
            plt.savefig(picture_path)
        else:
            for i in range(1, 10):
                if not os.path.exists(picture_path + "-" + str(i) + ".png"):
                    plt.savefig(picture_path + "-" + str(i) + ".png")
                    break
        plt.close(fig)


def create_cost_chart(scenario,
                      time_step: int,
                      destination: str,
                      weights,
                      traj=None):
    """
    Create a chart with costs according to the principles of ethics of risk.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        weights (Dict): Read from weights.json. Defaults to None.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.

    Returns:
        No return value.
    """
    if len(traj) > 0:

        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(11.69, 8.27)

        # plot weighed risk costs adding up from bayesian, equality,
        # maximin, and ego costs
        bayes_weighed = [i * weights["bayes"]
                         for i in traj[0].risk_dict["bayes"]]
        equality_weighed = [i * weights["equality"]
                            for i in traj[0].risk_dict["equality"]]
        maximin_weighed = [i * weights["maximin"]
                           for i in traj[0].risk_dict["maximin"]]
        ego_weighed = [i * weights["ego"] for i in traj[0].risk_dict["ego"]]

        ax1.plot(bayes_weighed, label="Weighed Bayesian Costs",
                 color="green", lw=1)
        ax1.plot(equality_weighed, label="Weighed Equality Costs",
                 color="yellow", lw=1)
        ax1.plot(maximin_weighed, label="Weighed Maximin Costs",
                 color="red", lw=1)
        ax1.plot(ego_weighed, label="Weighed Ego Costs", color="orange", lw=1)
        ax1.plot(traj[0].risk_dict["total_weighed"],
                 label="Total Weighed Risk Costs", color="blue", lw=2)

        # add description of harm plot
        ax1.legend(loc='upper right')
        ax1.set_ylabel("risk cost (time adjusted and weighed)")

        fig.suptitle("Risk costs")

        # Create directory for pictures
        destination = os.path.join(destination, str(scenario.benchmark_id))
        if not os.path.exists(destination):
            os.makedirs(destination)

        picture_path = destination + "/Costs_" + str(time_step)
        if not os.path.exists(picture_path + ".png"):
            plt.savefig(picture_path)
        else:
            for i in range(1, 10):
                if not os.path.exists(picture_path + "-" + str(i) + ".png"):
                    plt.savefig(picture_path + "-" + str(i) + ".png")
                    break
        plt.close(fig)


def create_total_cost_chart(scenario,
                            time_step: int,
                            destination: str,
                            risk_modes,
                            traj=None):
    """
    Create a chart with total risk costs for the most cost-efficient trajs.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.

    Returns:
        No return value.
    """
    if traj is not None:
        # check if enough trajectories are available to plot
        if risk_modes["figures"]["number_plotted_trajectories"] > len(traj):
            number = len(traj)
        else:
            number = risk_modes["figures"]["number_plotted_trajectories"]
    else:
        number = 0

    # check if valid Frenét trajectories exist
    if number > 0:

        # create fourth figure to display costs
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(11.69, 8.27)

        i = 0

        for ft in traj[0:number]:
            # plot risk costs
            ax1.plot(ft.risk_dict["bayes"], label="Trajectory " + str(i + 1),
                     color=col[i], lw=1)
            ax2.plot(ft.risk_dict["equality"],
                     label="Trajectory " + str(i + 1), color=col[i], lw=1)
            ax3.plot(ft.risk_dict["maximin"], label="Trajectory " + str(i + 1),
                     color=col[i], lw=1)
            ax4.plot(ft.risk_dict["ego"], label="Trajectory " + str(i + 1),
                     color=col[i], lw=1)
            i += 1

        fig.suptitle("Risk costs")
        ax1.set_ylabel("Bayesian Costs")
        ax2.set_ylabel("Equality Costs")
        ax3.set_ylabel("Maximin Costs")
        ax4.set_ylabel("Ego Costs")
        ax2.legend(loc='upper right')

        # Create directory for pictures
        destination = os.path.join(destination, str(scenario.benchmark_id))
        if not os.path.exists(destination):
            os.makedirs(destination)

        picture_path = destination + "/Traj_" + str(time_step)
        if not os.path.exists(picture_path + ".png"):
            plt.savefig(picture_path)
        else:
            for i in range(1, 10):
                if not os.path.exists(picture_path + "-" + str(i) + ".png"):
                    plt.savefig(picture_path + "-" + str(i) + ".png")
                    break
        plt.close(fig)
