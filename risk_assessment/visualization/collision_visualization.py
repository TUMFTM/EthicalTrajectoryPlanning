"""Function to create visualization of collision and according harm."""

import os
import matplotlib.pyplot as plt
import numpy as np
from commonroad.visualization.draw_dispatch_cr import draw_object


def collision_vis(scenario,
                  destination: str,
                  ego_harm: float,
                  ego_type,
                  ego_v: float,
                  ego_mass: float,
                  obs_harm: float,
                  obs_type,
                  obs_v: float,
                  obs_mass: float,
                  pdof: float,
                  ego_angle: float,
                  obs_angle: float,
                  time_step: int,
                  modes,
                  marked_vehicle: [int] = None,
                  planning_problem=None,
                  global_path: np.ndarray = None,
                  driven_traj=None):
    """
    Create a report for visualization of the collision and respective harm.

    Create a collision report and saves the file in the destination folder
    under a subfolder "collisions".

    Args:
        scenario (Scenario): Considered Scenario.
        destination (str): Path to save output.
        ego_type (Obstacle): Type of the ego vehicle (usually CAR).
        ego_harm (float): Harm for the ego vehicle.
        ego_v (float): Impact speed of the ego vehicle.
        ego_mass (float): Mass of the ego vehicle.
        obs_harm (float): Harm for the obstacle.
        obs_type (Obstacle): Type of obstacle.
        obs_v (float): Velocity of the obstacle.
        obs_mass (float): Estimated mass of the obstacle.
        pdof (float): Principle degree of force.
        ego_angle (float): Angle of impact area for the ego vehicle.
        obs_angle (float): Angle of impact area for the obstacle.
        time_step (int): Current time step.
        modes (Dict): Risk modes. Read from risk.json.
        marked_vehicle ([int]): IDs of the marked vehicles.
            Defaults to None.
        planning_problem (PlanningProblem): Considered planning
            problem. Defaults to None.
        global_path (np.ndarray): Global path for the planning problem.
            Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego
            vehicle. Defaults to None.

    Returns:
        No return value.
    """
    # clear everything
    plt.cla()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(5.833, 8.25)

    # set plot limits to show the road section around the ego vehicle
    position = scenario.obstacle_by_id(marked_vehicle).\
        occupancy_at_time(time_step=time_step).shape.center
    plot_limits = [position[0] - 20,
                   position[0] + 20,
                   position[1] - 20,
                   position[1] + 20]

    # plot the scenario at the current time step
    draw_object(scenario,
                draw_params={'time_begin': time_step, 'scenario':
                             {'dynamic_obstacle': {'show_label': True}}},
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

    # draw driven trajectory
    if driven_traj is not None:
        x = [state.position[0] for state in driven_traj]
        y = [state.position[1] for state in driven_traj]
        plt.plot(x, y, color='green', zorder=25)

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

    # get mode description
    if modes["harm_mode"] == "log_reg":
        mode = "logistic regression"
        harm = "P(MAIS 3+)"
    elif modes["harm_mode"] == "ref_speed":
        mode = "reference speed"
        harm = "P(MAIS 3+)"
    elif modes["harm_mode"] == "gidas":
        mode = "GIDAS P(MAIS 2+)"
        harm = "P(MAIS 2+)"
    else:
        mode = "None"

    # get angle mode
    if modes["ignore_angle"] is True or modes["harm_mode"] == "gidas":
        angle = "ignoring impact areas"
    else:
        if modes["reduced_angle_areas"]:
            angle = "considering impact areas reduced on front, side, and " \
                "rear crashes"
        else:
            angle = "considering impact areas according to the clock system"

        if modes["sym_angle"]:
            angle += " with symmetric coefficients"
        else:
            angle += " with asymmetric coefficients"

    # description of crash
    description = "Collision at {:.1f} s in ".\
        format(time_step * scenario.dt) + \
        str(scenario.benchmark_id) + \
        "\n\nCalculate harm using the " + mode + " model by " + angle + \
        "\n\nEgo vehicle harm " + harm + ": {:.3f}".format(ego_harm) + \
        "\nObstacle harm " + harm + ": {:.3f}".format(obs_harm) + \
        "\n\nCrash parameters:\n\nEgo type: " + str(ego_type)[13:] + \
        "\nEgo velocity: {:.2f}m/s". \
        format(ego_v) + \
        "\nEgo mass: {:.0f}kg".format(ego_mass) + \
        "\nImpact angle for the ego vehicle: {:.2f}°". \
        format(ego_angle * 180 / np.pi) + \
        "\n\nObstacle type: " + str(obs_type)[13:]
    if obs_mass is not None and obs_v is not None and obs_angle is not None:
        description += "\nObstacle velocity: {:.2f}m/s".format(obs_v) + \
            "\nObstacle mass: {:.0f}kg".format(obs_mass) + \
            "\nImpact angle for the obstacle: {:.2f}°". \
            format(obs_angle * 180 / np.pi)
    description += "\n\nPrinciple degree of force: {:.2f}°". \
        format(pdof * 180 / np.pi)

    # add description of crash
    ax1.axis('off')
    ax1.text(0, 1, description, verticalalignment='top', fontsize=8,
             wrap=True)

    fig.suptitle("Collision in " + str(scenario.benchmark_id) + " detected",
                 fontsize=16)

    # Create directory for pictures
    destination = os.path.join(destination, "collisions")
    if not os.path.exists(destination):
        os.makedirs(destination)

    plt.savefig(destination + "/" + str(scenario.benchmark_id) + ".png")
    plt.close()
