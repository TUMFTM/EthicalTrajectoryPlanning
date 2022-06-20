#!/user/bin/env python

"""Visualization functions for the frenét planner."""
import warnings

from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad_helper_functions.visualization import (
    get_max_frames_from_scenario,
    get_plot_limits_from_scenario,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import sys
import os

# Ignore Matplotlib DeprecationWarning
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

plt.rcParams["figure.figsize"] = (8, 8)

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)

from planner.Frenet.utils.helper_functions import (
    get_max_curvature,
    green_to_red_colormap,
)
from planner.utils.timers import ExecTimer
from prediction.utils.visualization import draw_uncertain_predictions

i = 0


def animate_scenario(
    scenario: Scenario,
    fps: int = 30,
    plot_limits: [float] = None,
    marked_vehicles: [int] = None,
    planning_problem=None,
    save_animation: bool = False,
    animation_directory: str = "./out/",
    animation_area: float = None,
    success: bool = None,
    failure_msg: str = None,
    exec_timer=None,
):
    """
    Animate a commonroad scenario.

    Args:
        scenario (Scenario): Scenario to be animated.
        fps (int): Frames per second. Defaults to 30.
        plot_limits ([float]): Plot limits for the scenario. Defaults to None.
        marked_vehicles ([int]): IDs of the vehicles that should be marked. Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem. Defaults to None.
        save_animation (bool): True if the animation should be saved. Defaults to False.
        animation_directory (str): Directory to save the animation in. Defaults to './out/'.
        animation_area (float): Size of the animated area). Defaults to None.
        success (bool): True if it is a successfully solved scenario. Defaults to None.
        failure_msg (str): Failure-message of the scenario. Defaults to None.
        exec_times_dict (dict): Dictionary with the execution times. Defaults to None.

    Returns:
        animation: Animated scenario.
        dict: Dictionary with the execution times.
    """
    # Create a dummy logger that does nothing if no timer is given
    if exec_timer is None:
        exec_timer = ExecTimer(False)
    # get the plot limits
    if plot_limits is None:
        plot_limits = get_plot_limits_from_scenario(scenario=scenario)

    # get the frames per second
    if 1 / fps < scenario.dt:
        fps_available = 1 / scenario.dt
    else:
        fps_available = fps

    # get the number of frames (number of time steps until the planning problem is solved
    if marked_vehicles is not None:
        frames = (
            len(scenario.obstacle_by_id(marked_vehicles[0]).prediction.occupancy_set)
            + 1
        )
    elif planning_problem is not None and hasattr(
        planning_problem.goal.state_list[0], "time_step"
    ):
        frames = planning_problem.goal.state_list[0].time_step.end + 1
    else:
        trajectory_points = get_max_frames_from_scenario(scenario=scenario)
        frames = int(trajectory_points * scenario.dt * fps_available)

    if frames == 0:
        frames = 1

    # get the states of the marked vehicle
    t = []
    v = []
    a = []
    yaw = []

    with exec_timer.time_with_cm("animate create states"):

        if marked_vehicles is not None:
            # add the initial states if they are given in the planning problme
            trajectory = scenario.obstacle_by_id(
                marked_vehicles[0]
            ).prediction.trajectory
            if hasattr(
                scenario.obstacle_by_id(marked_vehicles[0]).initial_state, "time_step"
            ):
                t.append(
                    scenario.obstacle_by_id(marked_vehicles[0]).initial_state.time_step
                )
            else:
                t.append(0)
            if hasattr(
                scenario.obstacle_by_id(marked_vehicles[0]).initial_state, "velocity"
            ):
                v.append(
                    scenario.obstacle_by_id(marked_vehicles[0]).initial_state.velocity
                )
            else:
                v.append(0.0)
            if hasattr(
                scenario.obstacle_by_id(marked_vehicles[0]).initial_state,
                "acceleration",
            ):
                a.append(
                    scenario.obstacle_by_id(
                        marked_vehicles[0]
                    ).initial_state.acceleration
                )
            else:
                a.append(0.0)
            if hasattr(
                scenario.obstacle_by_id(marked_vehicles[0]).initial_state, "orientation"
            ):
                yaw.append(
                    scenario.obstacle_by_id(
                        marked_vehicles[0]
                    ).initial_state.orientation
                )
            else:
                yaw.append(0.0)

            # add every state from the trajectory
            for state in trajectory.state_list:
                if hasattr(state, "velocity"):
                    v.append(state.velocity)
                else:
                    v.append(0.0)
                if hasattr(state, "time_step"):
                    t.append(state.time_step)
                else:
                    t.append(0)
                if hasattr(state, "acceleration"):
                    a.append(state.acceleration)
                else:
                    a.append(0.0)
                if hasattr(state, "orientation"):
                    yaw.append(state.orientation)
                else:
                    yaw.append(0.0)

    # get information about the success of the solved scenario
    # there are 2 directories, one for successful scenarios and one for failed ones
    # create these directories if they do not exist yet
    success_or_not = ""

    if success is not None:
        if success is True:
            success_or_not = "Succeeded!"
            if failure_msg is not None:
                success_or_not += "\n" + failure_msg
            animation_directory = animation_directory + "/succeeded/"
            if not os.path.exists(animation_directory):
                os.makedirs(animation_directory)
        else:
            success_or_not = "Failed!"
            if failure_msg is not None:
                success_or_not += "\n" + failure_msg
            animation_directory = animation_directory + "/failed/"
            if not os.path.exists(animation_directory):
                os.makedirs(animation_directory)

    def animate(j):

        # axis 1 sows the marked vehicle in the lanelet network
        ax1.cla()
        if hasattr(planning_problem.goal.state_list[0], "time_step"):
            target_time_string = "Target-time: %.1f s - %.1f s" % (
                planning_problem.goal.state_list[0].time_step.start * scenario.dt,
                planning_problem.goal.state_list[0].time_step.end * scenario.dt,
            )
        else:
            target_time_string = "No target-time"
        ax1.set(
            title=(
                str(scenario.benchmark_id)
                + ": "
                + success_or_not
                + "\n\nTime: "
                + str(round(j * scenario.dt, 1))
                + " s\n"
                + target_time_string
            )
        )
        ax1.set_aspect("equal")
        ax1.set_xlabel(r"$x$ in m")
        ax1.set_ylabel(r"$y$ in m")
        # draw all obstacles
        draw_object(
            obj=scenario,
            ax=ax1,
            plot_limits=plot_limits,
            draw_params={"time_begin": int(j / (scenario.dt * fps_available))},
        )

        # draw the planning problme
        if planning_problem is not None:
            draw_object(
                obj=planning_problem,
                ax=ax1,
                plot_limits=plot_limits,
                draw_params={"time_begin": int(j / (scenario.dt * fps_available))},
            )

        # draw the ego vehicle in green
        # and put the ego vehicle in the center of the plot
        if marked_vehicles is not None:
            for marked_vehicle in marked_vehicles:
                if marked_vehicle is not None:
                    draw_object(
                        obj=scenario.obstacle_by_id(marked_vehicle),
                        ax=ax1,
                        plot_limits=plot_limits,
                        draw_params={
                            "time_begin": int(j / (scenario.dt * fps_available)),
                            "facecolor": "g",
                        },
                    )
                    if animation_area is not None:
                        # align ego position to the center
                        ego_vehicle = scenario.obstacle_by_id(marked_vehicle)
                        if j == 0:
                            ego_vehicle_pos = ego_vehicle.initial_state.position
                        else:
                            ego_vehicle_pos = ego_vehicle.prediction.occupancy_set[
                                j - 1
                            ].shape.center
                        ax1.set(
                            xlim=(
                                ego_vehicle_pos[0] - animation_area,
                                ego_vehicle_pos[0] + animation_area,
                            )
                        )
                        ax1.set(
                            ylim=(
                                ego_vehicle_pos[1] - animation_area,
                                ego_vehicle_pos[1] + animation_area,
                            )
                        )

                # velocity subplot
                ax2.cla()
                ax2.set(title="Velocity")

                ax2.set(ylabel=r"$v$ in m/s")
                ax2.set(xlabel=r"$t$ in s")
                # visualize the given goal velocity in the planning problem
                if hasattr(planning_problem.goal.state_list[0], "velocity"):
                    v_min = planning_problem.goal.state_list[0].velocity.start
                    v_max = planning_problem.goal.state_list[0].velocity.end
                    if hasattr(planning_problem.goal.state_list[0], "time_step"):
                        ts_min = planning_problem.goal.state_list[0].time_step.start
                        ts_max = planning_problem.goal.state_list[0].time_step.end
                        ax2.plot(
                            [ts_min, ts_max, ts_max, ts_min, ts_min],
                            [v_min, v_min, v_max, v_max, v_min],
                            color="g",
                            label="goal area",
                        )
                    else:
                        ax2.plot([t[0], t[-1]], [v_min, v_min], color="g")
                        ax2.plot(
                            [t[0], t[-1]], [v_max, v_max], color="g", label="goal area"
                        )
                    ax2.legend()
                ax2.plot(t, v)
                ax2.scatter(j, v[j])

                # acceleration subplot
                ax3.cla()
                ax3.set(title="Acceleration")
                ax3.set(ylabel=r"$a$ in m/s²")
                ax3.set(xlabel=r"$t$ in s")
                # visualize the given goal acceleration in the planning problem
                if hasattr(planning_problem.goal.state_list[0], "acceleration"):
                    a_min = planning_problem.goal.state_list[0].acceleration.start
                    a_max = planning_problem.goal.state_list[0].acceleration.end
                    if hasattr(planning_problem.goal.state_list[0], "time_step"):
                        ts_min = planning_problem.goal.state_list[0].time_step.start
                        ts_max = planning_problem.goal.state_list[0].time_step.end
                        ax3.plot(
                            [ts_min, ts_max, ts_max, ts_min, ts_min],
                            [a_min, a_min, a_max, a_max, a_min],
                            color="g",
                            label="goal area",
                        )
                    else:
                        ax3.plot([t[0], t[-1]], [a_min, a_min], color="g")
                        ax3.plot(
                            [t[0], t[-1]], [a_max, a_max], color="g", label="goal area"
                        )
                    ax3.legend()
                ax3.plot(t, a)
                ax3.scatter(j, a[j])

                # orientation subplot
                ax4.cla()
                ax4.set(title="Orientation")
                ax4.set(ylabel=r"$\psi$ in rad")
                ax4.set(xlabel=r"$t$ in s")
                # visualize the given goal orientation in the planning problem
                if hasattr(planning_problem.goal.state_list[0], "orientation"):
                    yaw_min = planning_problem.goal.state_list[0].orientation.start
                    yaw_max = planning_problem.goal.state_list[0].orientation.end
                    if hasattr(planning_problem.goal.state_list[0], "time_step"):
                        ts_min = planning_problem.goal.state_list[0].time_step.start
                        ts_max = planning_problem.goal.state_list[0].time_step.end
                        ax4.plot(
                            [ts_min, ts_max, ts_max, ts_min, ts_min],
                            [yaw_min, yaw_min, yaw_max, yaw_max, yaw_min],
                            color="g",
                            label="goal area",
                        )
                    else:
                        ax4.plot([t[0], t[-1]], [yaw_min, yaw_min], color="g")
                        ax4.plot(
                            [t[0], t[-1]],
                            [yaw_max, yaw_max],
                            color="g",
                            label="goal area",
                        )
                    ax4.legend()
                ax4.plot(t, yaw)
                ax4.scatter(j, yaw[j])

    plt.close()
    # create the figure
    fig = plt.figure(constrained_layout=False, figsize=(22, 15))
    # set the font size
    plt.rcParams.update({"font.size": 25})
    # add a gridspec for the subplots
    gs = fig.add_gridspec(3, 11, left=0.05, top=0.9, right=0.95, wspace=0.3, hspace=0.5)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2, 0:3])
    ax3 = fig.add_subplot(gs[2, 4:7])
    ax4 = fig.add_subplot(gs[2, 8:11])

    with exec_timer.time_with_cm("animate create animation"):
        anim = animation.FuncAnimation(
            fig=fig,
            func=animate,
            frames=frames,
            interval=1 / fps_available * 1000,
            repeat=False,
            repeat_delay=1000,
            blit=False,
        )

    # save the animation
    with exec_timer.time_with_cm("animate save"):
        if save_animation:
            writergif = animation.PillowWriter(fps=fps_available)
            anim.save(
                animation_directory + scenario.benchmark_id + ".gif", writer=writergif
            )
    return anim


def draw_frenet_trajectories(
    scenario,
    time_step: int,
    marked_vehicle: [int] = None,
    planning_problem=None,
    traj=None,
    all_traj=None,
    predictions: dict = None,
    visible_area=None,
    animation_area: float = 40.0,
    global_path: np.ndarray = None,
    global_path_after_goal: np.ndarray = None,
    driven_traj=None,
    ax=None,
    picker=False,
    show_label=False,
    live=True,
):
    """
    Plot all frenét trajectories.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        marked_vehicle ([int]): IDs of the marked vehicles. Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem. Defaults to None.
        traj (FrenetTrajectory): The best trajectory of all frenét trajectories. Defaults to None.
        all_traj ([FrenetTrajectory]): All frenét trajectories. Defaults to None.
        fut_pos_list (np.ndarray): Future positions of the vehicles. Defaults to None.
        visible_area (shapely.Polygon): Polygon of the visible area. Defaults to None.
        animation_area (float): Area that should be shown. Defaults to 40.0.
        global_path (np.ndarray): Global path for the planning problem. Defaults to None.
        global_path_after_goal (np.ndarray): Global path for the planning problem after reaching the goal. Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego vehicle. Defaults to None.
        save_fig (bool): True if the figure should be saved. Defaults to False.
    """
    if live:
        ax = draw_scenario(
            scenario,
            time_step,
            marked_vehicle,
            planning_problem,
            traj,
            visible_area,
            animation_area,
            global_path,
            global_path_after_goal,
            driven_traj,
            ax,
            picker,
            show_label,
        )

    # Draw all possible trajectories with their costs as colors
    if all_traj is not None:

        # x and y axis description
        ax.set_xlabel("x in m")
        ax.set_ylabel("y in m")

        # align ego position to the center
        ax.set_xlim(
            all_traj[0].x[0] - animation_area, all_traj[0].x[0] + animation_area
        )
        ax.set_ylim(
            all_traj[0].y[0] - animation_area, all_traj[0].y[0] + animation_area
        )

        # mormalize the costs of the trajectories to map colors to them
        norm = matplotlib.colors.Normalize(
            vmin=min([all_traj[i].cost for i in range(len(all_traj))]),
            vmax=max([all_traj[i].cost for i in range(len(all_traj))]),
            clip=True,
        )
        mapper = cm.ScalarMappable(norm=norm, cmap=green_to_red_colormap())

        # first plot all invalid trajectories
        for p in all_traj:
            if p.valid_level < 1:
                ax.plot(
                    p.x,
                    p.y,
                    alpha=0.4,
                    color=(0.7, 0.7, 0.7),
                    zorder=19,
                    picker=picker,
                )
            elif p.valid_level < 10:
                ax.plot(
                    p.x,
                    p.y,
                    alpha=0.6,
                    color=(0.3, 0.3, 0.7),
                    zorder=20,
                    picker=picker,
                )

        # then plot all valid trajectories
        for p in reversed(all_traj):
            if p.valid_level >= 10:
                color = mapper.to_rgba(p.cost)
                ax.plot(p.x, p.y, alpha=1.0, color=color, zorder=20, picker=picker)

    # draw planned trajectory
    if traj is not None:
        ax.plot(
            traj.x,
            traj.y,
            alpha=1.0,
            color="green",
            zorder=25,
            lw=3.0,
            label="Best trajectory",
            picker=picker,
        )

    # draw predictions
    if predictions is not None:
        draw_uncertain_predictions(predictions, ax)

    # show the figure until the next one ins ready
    # plt.savefig(str(i).zfill(4) + ".png")
    # i += 1
    plt.pause(0.0001)


def show_frenet_details(vehicle_params, fp_list, global_path: np.ndarray = None):
    """
    Plot details about the frenét trajectories.

    Args:
        vehicle_params (VehicleParameters): Parameters of the ego vehicle.
        fp_list ([FrenetTrajectory]): Considered frenét trajectories.
        global_path (np.ndarray): Global path of the planning problem. Defaults to None
    """
    # create the figure
    fig = plt.figure(constrained_layout=False, figsize=(17, 10))
    # set the font size
    plt.rcParams.update({"font.size": 15})
    # add a gridspec for the subplots
    gs = fig.add_gridspec(3, 2, left=0.05, top=0.9, right=0.95, wspace=0.3, hspace=0.5)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])

    # plot the frenét paths
    for fp in fp_list:
        if fp.valid >= 10:
            col = "g"
        else:
            col = "r"
        ax1.plot(fp.x, fp.y, color=col)

    ax1.set_aspect("equal")
    ax1.set_title("Global trajectory")
    ax1.set_xlabel(r"$x$ in m")
    ax1.set_ylabel(r"$y$ in m")

    if global_path is not None:
        ax1.plot(global_path[:, 0], global_path[:, 1], color="b")

    # curvature
    ax2.set_title("Curvature")
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_ylabel(r"$\kappa$ in rad/m")
    ax2.set_xlabel(r"$t$ in s")
    for fp in fp_list:
        ax2.plot(fp.t, fp.curv)
        max_curv = []
        for i in range(len(fp.t)):
            max_curv_i, _ = get_max_curvature(vehicle_params=vehicle_params, v=fp.v[i])
            max_curv.append(abs(max_curv_i))
        ax2.plot(fp.t, max_curv, color="r")
        ax2.plot(fp.t, np.multiply((-1), max_curv), color="r")

    # lateral offset
    ax3.set_title("Lateral offset")
    ax3.set_ylabel(r"$d$ in m")
    ax3.set_xlabel(r"$t$ in s")
    for fp in fp_list:
        ax3.plot(fp.t, fp.d)

    # covered arc length
    ax4.set_title("Covered arc length")
    ax4.set_ylabel(r"$s$ in m")
    ax4.set_xlabel(r"$t$ in s")
    for fp in fp_list:
        ax4.plot(fp.t, fp.s)

    plt.show()


def draw_reach_sets(
    traj=None,
    animation_area: float = 55.0,
    reach_set=None,
    ax=None,
):
    """
    Plot reachable sets.

    Plot reachable sets of all objects except ego.

    Args:
        traj (FrenetTrajectory): The best trajectory of all frenét trajectories. Defaults to None.
        animation_area (float): Area that should be shown. Defaults to 55.0.
        ax (Axes): Plot.
    """
    # draw reach sets
    if reach_set is not None:
        for idx in reach_set:
            no_sets = len(reach_set[idx])
            set_nr = 0
            for reach_set_of_id in reach_set[idx]:
                set_nr += 1
                for reach_set_step in reach_set_of_id.keys():
                    polygon = Polygon(
                        reach_set_of_id[reach_set_step],
                        closed=True,
                        alpha=0.075,
                        color="blue",
                        fill=True,
                        label="reach_set "
                        + str(set_nr)
                        + "/"
                        + str(no_sets)
                        + " of ID "
                        + str(idx)
                        + " , step = "
                        + str(reach_set_step),
                        zorder=25,
                        lw=0,  # line width zero to hide seam from exterior to interior
                    )
                    ax.add_patch(polygon)

        # align ego position to the center
        ax.set_xlim(traj.x[0] - animation_area, traj.x[0] + animation_area)
        ax.set_ylim(traj.y[0] - animation_area, traj.y[0] + animation_area)


def draw_scenario(
    scenario: Scenario = None,
    time_step: int = 0,
    marked_vehicle=None,
    planning_problem=None,
    traj=None,
    visible_area=None,
    animation_area: float = 55.0,
    global_path: np.ndarray = None,
    global_path_after_goal: np.ndarray = None,
    driven_traj=None,
    ax=None,
    picker=False,
    show_label=False,
):
    """
    Draw scenario.

    General drawing function for scenario.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        marked_vehicle ([int]): IDs of the marked vehicles. Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem. Defaults to None.
        traj (FrenetTrajectory): The best trajectory of all frenét trajectories. Defaults to None.
        visible_area (shapely.Polygon): Polygon of the visible area. Defaults to None.
        animation_area (float): Area that should be shown. Defaults to 40.0.
        global_path (np.ndarray): Global path for the planning problem. Defaults to None.
        global_path_after_goal (np.ndarray): Global path for the planning problem after reaching the goal. Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego vehicle. Defaults to None.
        ax (Axes): Plot.

    Returns:
        Axes: Plot with scenario.
    """
    # clear everything
    global i
    if ax is None:
        ax = plt.subplot()
    ax.cla()
    # plot the scenario at the current time step
    draw_object(
        scenario,
        draw_params={
            "time_begin": time_step,
            "dynamic_obstacle": {
                "draw_shape": True,
                "draw_bounding_box": True,
                "draw_icon": False,
                "show_label": show_label,
            },
        },
        ax=ax,
    )
    ax.set_aspect("equal")

    # draw the planning problem
    if planning_problem is not None:
        draw_object(planning_problem, ax=ax)

    if marked_vehicle is not None:
        # mark the ego vehicle
        draw_object(
            obj=scenario.obstacle_by_id(marked_vehicle),
            draw_params={
                "time_begin": time_step,
                "facecolor": "g",
                "dynamic_obstacle": {
                    "draw_shape": False,
                    "draw_bounding_box": False,
                    "draw_icon": True,
                },
            },
        )

    # Draw global path
    if global_path is not None:
        ax.plot(
            global_path[:, 0],
            global_path[:, 1],
            color="blue",
            zorder=20,
            label="Global path",
        )
        if global_path_after_goal is not None:
            ax.plot(
                global_path_after_goal[:, 0],
                global_path_after_goal[:, 1],
                color="blue",
                zorder=20,
                linestyle="--",
            )

    # draw driven trajectory
    if driven_traj is not None:
        x = [state.position[0] for state in driven_traj]
        y = [state.position[1] for state in driven_traj]
        ax.plot(x, y, color="green", zorder=25, label="Driven trajectory")

    # draw visible sensor area
    if visible_area is not None:
        if visible_area.geom_type == "MultiPolygon":
            for geom in visible_area.geoms:
                ax.fill(*geom.exterior.xy, "g", alpha=0.2, zorder=10)
        elif visible_area.geom_type == "Polygon":
            ax.fill(*visible_area.exterior.xy, "g", alpha=0.2, zorder=10)
        else:
            for obj in visible_area:
                if obj.geom_type == "Polygon":
                    ax.fill(*obj.exterior.xy, "g", alpha=0.2, zorder=10)

    # get the target time to show it in the title
    if hasattr(planning_problem.goal.state_list[0], "time_step"):
        target_time_string = "Target-time: %.1f s - %.1f s" % (
            planning_problem.goal.state_list[0].time_step.start * scenario.dt,
            planning_problem.goal.state_list[0].time_step.end * scenario.dt,
        )
    else:
        target_time_string = "No target-time"

    # ax.legend()
    ax.set_title(
        "Time: {0:.1f} s".format(time_step * scenario.dt) + "    " + target_time_string
    )
    return ax


# EOF
