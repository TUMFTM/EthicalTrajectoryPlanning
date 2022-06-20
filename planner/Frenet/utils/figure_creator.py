#!/user/bin/env python

"""Tools to visualize scenarios or to create figures."""

import matplotlib.pyplot as plt
import os
import sys
import csv
import numpy as np
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.planning.goal import GoalRegion, Interval
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_dc.collision.visualization.draw_dispatch import draw_object

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)

from planner.Frenet.utils.helper_functions import TUMColors
from planner.Frenet.utils.timing_helpers import get_processes_and_subprocesses


def get_plotting_data(general_key: str, sub_keys: [str], times_dict: dict):
    """
    Get the data from the execution times dictionary.

    Args:
        general_key (str): Name of the general function.
        sub_keys ([str]): Names of the subfunctions.
        times_dict (dict): Dictionary with the execution times.

    Returns:
        [float]: Total times shown in the pie chart.
        [str]: Keys of the times shown in the pie chart.
        [str]: Data for the table.
    """
    # get the time of the general function
    total_time = times_dict[general_key][0]

    keys = sub_keys
    total_times = []
    avg_times = []
    n_execs = []

    # get total times, average times and number of executions from the subfunctions
    for key in keys:
        total_times.append(times_dict[key][0])
        avg_times.append(times_dict[key][2])
        n_execs.append(times_dict[key][1])

    # get information about other stuff in the function that happens between subfunctions and is not times individually
    keys.append('other stuff')
    total_times.append(max(0.0, (total_time - np.sum(total_times))))
    n_execs.append(times_dict[general_key][1])
    avg_times.append(total_times[-1] / n_execs[-1])

    # only show functions that use more than 1 % of the time, else put them into other stuff
    # for better understanding of the final plot
    shown_total_times = []
    shown_keys = []
    total_time_small_stuff = 0.0

    # sum up the other stuff
    for i in range(len(keys)):
        if total_times[i] < total_time / 100.0 or keys[i] == 'other stuff':
            total_time_small_stuff += total_times[i]
        else:
            shown_total_times.append(total_times[i])
            shown_keys.append(keys[i])

    shown_keys.append('other stuff')
    shown_total_times.append(total_time_small_stuff)

    return shown_total_times, shown_keys


def create_table(
    path: str,
    col_labels: [str],
    times_dict: dict,
    processes_and_subprocesses: dict,
    key: str,
):
    """
    Create and save a table that shows the execution times of a process and its subprocesses.

    Args:
        path (str): Where to save the table.
        col_labels ([str]): Labels for the columns of the table.
        times_dict (dict): Dictionary with the execution times.
        processes_and_subprocesses (dict): Dictionary with all processes and subprocesses.
        key (str): Name of the process to be evaluated.
    """
    # create the table
    log_file = open(path + key + ' table.csv', 'w')

    # add the column labels
    for label in col_labels:
        log_file.write(label + ';')
    log_file.write('\n')

    # get the number of executions
    if key == 'total time':
        n_execs = times_dict['read scenario'][1]
    else:
        n_execs = times_dict[key][1]

    # get the execution time of the evaluated process
    key_time = times_dict[key][0]

    # write the information about the process
    log_file.write(
        key
        + ';'
        + str(round(times_dict[key][0] / n_execs, 5))
        + ';'
        + str(int(times_dict[key][1] / n_execs))
        + ';'
        + str(round(times_dict[key][2], 5))
        + ';'
        + str(round(100 * times_dict[key][0] / key_time, 2))
        + '\n'
    )

    # write the information about the subprocesses
    for subkey in list(processes_and_subprocesses[key]):
        log_file.write(
            subkey
            + ';'
            + str(round(times_dict[subkey][0] / n_execs, 5))
            + ';'
            + str(int(times_dict[subkey][1] / n_execs))
            + ';'
            + str(round(times_dict[subkey][2], 5))
            + ';'
            + str(round(100 * times_dict[subkey][0] / key_time, 2))
            + '\n'
        )

    # close the file
    log_file.close()


def create_csv_tables(
    processes_and_subprocesses: dict,
    times_dict: dict,
    path: str,
    keys_to_create_tables: [str] = [
        'total time',
        'simulation',
        'calculate trajectories',
        'sort trajectories',
        'calculate costs',
    ],
):
    """
    Create tables with information about the execution times for various processes.

    Args:
        processes_and_subprocesses (dict): Dictionary with the processes and subprocesses.
        times_dict (dict): Dictionary with the execution times.
        path (path): Where to save the tables.
        keys_to_create_tables ([str]): Names of the processes that should be evaluated. Defaults to ['total time', 'simulation', 'calculate trajectories', 'sort trajectories', 'calculate costs'].
    """
    # labels of the columns
    col_labels = [
        'process',
        'total time in s',
        'number of executions',
        'average time in s',
        '%',
    ]

    # create a table for every listed process
    for key in keys_to_create_tables:
        create_table(
            path=path,
            col_labels=col_labels,
            key=key,
            processes_and_subprocesses=processes_and_subprocesses,
            times_dict=times_dict,
        )


def create_bar_chart_and_tables(
    fontsize: float = 15.0,
    save_animation: bool = False,
    batch: str = '0-1000mode0',
    only_show_one: str = None,
):
    """
    Create and save a bar chart that visualizes the distribution of the execution times and save tables that hold information about the execution times.

    Args:
        fontsize (float): Fontsize of the plot. Defaults to 15.
        save_animation (bool): True if the animation was saved and this process also needs to be considered. Defaults to False.
        batch (str): Name of the batch that should be evaluated. Defaults to '0-1000mode0'.
        only_show_one (str): Name of the function that should be evaluated if only one function should be evaluated.
    """
    # open the txt file and parse it
    times_dict = {}
    path = 'planner/Frenet/results/batches/' + batch + '/'
    with open(path + 'execution_times.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                times_dict[row[0]] = [float(row[1]), float(row[2]), float(row[3])]
                line_count += 1

    # get the names of the processes and subprocesses used when timing the planner
    processes_and_subprocesses = get_processes_and_subprocesses(
        with_animation=save_animation
    )

    # create csv tables for the functions
    create_csv_tables(
        processes_and_subprocesses=processes_and_subprocesses,
        times_dict=times_dict,
        path=path,
    )

    plt.close()

    # update fontsize
    plt.rcParams.update({'font.size': fontsize})

    # create array with colors for bars
    colors = [
        TUMColors.tum_blue.value,
        TUMColors.tum_orange.value,
        TUMColors.tum_green.value,
        TUMColors.tum_gray2.value,
        TUMColors.tum_blue.value,
        TUMColors.tum_orange.value,
        TUMColors.tum_green.value,
        TUMColors.tum_gray2.value,
        TUMColors.tum_blue.value,
        TUMColors.tum_orange.value,
        TUMColors.tum_green.value,
        TUMColors.tum_gray2.value,
    ]

    # create array with hatches for bars
    hatches = [None, '//', '.', '+', '//', '.', '+', None, '.', '+', None, '//']

    # select the keys that should be shown in the par chart
    if only_show_one is None:
        keys_to_be_shown = [
            'total time',
            'simulation',
            'calculate trajectories',
            'sort trajectories',
        ]
    else:
        keys_to_be_shown = [only_show_one]
    n_keys = len(keys_to_be_shown)

    # create the figure
    if only_show_one is None:
        plt.figure(figsize=(20, 20))
    else:
        plt.figure(figsize=(20, 5))

    # get the time that represents 100 % (simulation process) and the maximum time
    total_time = times_dict['simulation'][0]
    max_time = times_dict['total time'][0]
    if only_show_one is not None:
        total_time = times_dict[only_show_one][0]
        max_time = times_dict[only_show_one][0]

    # calculate the maximum percentage
    max_percent = max_time / total_time * 100

    # array with the locations of the legends for every bar
    legend_loc = np.linspace(1.0, 0.0, n_keys + 1)
    legend_loc = legend_loc[1:]
    if n_keys == 1:
        legend_loc = [0.5]
    if n_keys == 4:
        legend_loc = [0.89, 0.89 - 0.78 / 3, 0.11 + 0.78 / 3, 0.11]

    # create a bar for every process
    for i, key in enumerate(keys_to_be_shown):
        # get the data vor the bar
        times, labels = get_plotting_data(
            general_key=key,
            sub_keys=processes_and_subprocesses[key],
            times_dict=times_dict,
        )
        percent = [time / total_time * 100 for time in times]
        bottom = 0
        patches = []

        # plot the stacked bars
        for j in range(len(percent)):
            if only_show_one is None or only_show_one == key:
                if only_show_one is None:
                    height = 0.5
                else:
                    height = 0.2
                bar = plt.barh(
                    i,
                    percent[j],
                    left=bottom,
                    color=colors[j],
                    hatch=hatches[j],
                    height=height,
                    zorder=10,
                )
                bottom += percent[j]
                patches.append(bar[0])

        # create new labels that also mention the percentage
        new_labels = ['{0:1.1f} %: {1}'.format(i, j) for i, j in zip(percent, labels)]
        legend = plt.legend(
            handles=patches,
            labels=new_labels,
            bbox_to_anchor=(bottom / max_percent, legend_loc[i]),
            loc='center left',
        )
        plt.gca().add_artist(legend)

    # plot the ticks
    plt.yticks(list(range(len(keys_to_be_shown))), keys_to_be_shown)

    # set axis limits
    if only_show_one is not None:
        plt.ylim(-0.3, 0.3)

    # make the plot a bit more beautiful
    plt.gca().invert_yaxis()
    # hide frame
    right_side = plt.gca().spines['right']
    right_side.set_visible(False)
    top = plt.gca().spines['top']
    top.set_visible(False)
    # set grid
    plt.gca().xaxis.grid(True, zorder=5)
    plt.xlabel('execution time in %')

    # adjust subplots
    if only_show_one is None:
        plt.subplots_adjust(right=0.73, left=0.20)
    else:
        plt.subplots_adjust(right=0.68, left=0.23, bottom=0.2, top=0.97)

    plt.savefig(fname=(path + 'bar_chart'), format='pdf')
    plt.show()


def show_solution_trajectory_in_one_plot(
    fontsize: float = 15.0,
    n_shown_states: int = 5,
    mode: int = 0,
    result: str = 'failure',
    scenario_name: str = 'ZAM_Merge-1_1_T-1.xml',
):
    """
    Create a plot that shows the trajectory of the ego vehicle and the course of its velocity and orientation from a xml file that holds the solution.

    Args:
        fontsize (float): Fontsize of the plot. Defaults to 15.
        n_shown_states (int): Number of states shown in the plot. Defaults to 5.
        mode (int): Mode that was used to create the solution. Defaults to 0.
        result (str): 'failure' or 'success'. Defaults to 'success'.
        scenario_name (str): Name of the scenario to be evaluated. Defaults to 'ZAM_Merge-1_1_T-1.xml'.
    """
    # create the file path
    file_path = os.path.join(
        os.getcwd(),
        'planner/Frenet/results/solution trajectories/mode' + str(mode),
        result,
        scenario_name,
    )

    # set the fontsize of the plot
    plt.rcParams.update({'font.size': fontsize})

    # read the scenario, planning problem and ID of the ego vehicle
    crfr = CommonRoadFileReader(file_path)
    scenario, planning_problem_set = crfr.open()
    ego_id = int(crfr._get_author())

    # get the states of the ego obstacle
    ego_obst = scenario.obstacle_by_id(ego_id)
    ego_traj = ego_obst.prediction.trajectory.state_list
    x = [state.position[0] for state in ego_traj]
    y = [state.position[1] for state in ego_traj]
    v = [state.velocity for state in ego_traj]
    t = [state.time_step / 10 for state in ego_traj]
    orientation = [state.orientation for state in ego_traj]

    # create the plot with gridspec (3 subplots)
    fig = plt.figure(constrained_layout=False, figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, left=0.1, right=0.95, wspace=0.4, hspace=0.5)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])

    # plot the scenario in axis 1
    draw_object(scenario, ax=ax1)
    draw_object(planning_problem_set, ax=ax1)
    draw_object(
        obj=scenario.obstacle_by_id(ego_id),
        ax=ax1,
        draw_params={'facecolor': 'g', 'edgecolor': 'g'},
    )

    # get the time steps that are shown
    showed_ts = np.linspace(0, len(t) - 1, n_shown_states)
    showed_timesteps = [int(ts) for ts in showed_ts]

    # create an array with decreasing opacities
    opacities = np.linspace(1.0, 0.2, n_shown_states)

    # plot the scenario with the corresponding opacity for every shown timestep
    for i, timestep in enumerate(showed_timesteps):
        draw_object(
            obj=scenario.obstacle_by_id(ego_id),
            ax=ax1,
            draw_params={
                'time_begin': timestep,
                'facecolor': 'g',
                'edgecolor': 'g',
                'opacity': opacities[i],
            },
        )
        for obstacle in scenario.dynamic_obstacles:
            if obstacle.obstacle_id != ego_id:
                draw_object(
                    obj=obstacle,
                    ax=ax1,
                    draw_params={'time_begin': timestep, 'opacity': opacities[i]},
                )

    # plot the trajectory of the ego vehicle
    ax1.plot(x, y, color='g', zorder=25)

    # plot the trajectory of every obstacle
    for obstacle in scenario.dynamic_obstacles:
        if obstacle.obstacle_id != ego_id:
            obst_x = [
                state.position[0] for state in obstacle.prediction.trajectory.state_list
            ]
            obst_y = [
                state.position[1] for state in obstacle.prediction.trajectory.state_list
            ]
            obst_x.insert(0, obstacle.initial_state.position[0])
            obst_y.insert(0, obstacle.initial_state.position[1])
            ax1.plot(obst_x, obst_y, zorder=25, color='#1d7eea')

    # set the axis limit to focus on the ego vehicle
    ax1.set_xlim(min(x) - 10, max(x) + 10)
    ax1.set_ylim(min(y) - 10, max(y) + 10)

    ax1.set(xlabel=r'$x$ in m', ylabel=r'$y$ in m')
    ax1.set_aspect('equal')

    # create the velocity plot
    ax2.plot(t, v, color='g')
    ax2.set(title='velocity', xlabel=r'$t$ in s', ylabel=r'$v$ in m/s')
    ax2.set_xlim(min(t), max(t))

    # create the orientation plot
    ax3.plot(t, orientation, color='g')
    ax3.set(title='orientation', xlabel=r'$t$ in s', ylabel=r'$\psi$ in rad')
    ax3.set_xlim(min(t), max(t))

    plt.show()


def show_solution_trajectory_in_four_plots(
    fontsize: float = 15.0,
    mode: int = 0,
    result: str = 'success',
    scenario_name: str = 'RUS_Bicycle-4_1_T-1.xml',
):
    """
    Create a figure with four subplots to visualize the solution of a planning problem taken from a xml file.

    Scenarios:
        mode=0, result='success', scenario_name='RUS_Bicycle-4_1_T-1.xml'
        mode=2, result='success', scenario_name='USA_Peach-4_1_T-1.xml'

    Args:
        fontsize (float): Fontsize of the figure. Defaults to 15.
        mode (int): Mode used to solve the scenario. Defaults to 0.
        result (str): 'failure' or 'success'. Defaults to 'success'.
        scenario_name (str): Name of the scenario to be visualized. Defaults to 'RUS_Bicycle-4_1_T-1.xml'.
    """
    # create the file path of the scenario
    file_path = os.path.join(
        os.getcwd(),
        'planner/Frenet/results/solution trajectories/mode' + str(mode),
        result,
        scenario_name,
    )

    # set the fontsize
    plt.rcParams.update({'font.size': fontsize})

    # read scenario, planning problem and ID of the ego vehicle
    crfr = CommonRoadFileReader(file_path)
    scenario, planning_problem_set = crfr.open()
    ego_id = int(crfr._get_author())

    # get the trajectory of the ego vehicle
    ego_obst = scenario.obstacle_by_id(ego_id)
    ego_traj = ego_obst.prediction.trajectory.state_list
    x = [state.position[0] for state in ego_traj]
    y = [state.position[1] for state in ego_traj]
    t = [state.time_step / 10 for state in ego_traj]

    # create the figure with four subplots
    fig = plt.figure(constrained_layout=False, figsize=(12, 7))
    gs = fig.add_gridspec(9, 2)
    ax1 = fig.add_subplot(gs[:4, 0])
    ax2 = fig.add_subplot(gs[:4, 1])
    ax3 = fig.add_subplot(gs[5:, 0])
    ax4 = fig.add_subplot(gs[5:, 1])

    # get all available axes
    axes = [ax1, ax2, ax3, ax4]

    # get the time steps that are shown
    shown_ts = np.linspace(15, len(t) - 2, 4)
    shown_timesteps = [int(ts) for ts in shown_ts]

    # create the subplot for every axis
    for i, axis in enumerate(axes):
        # draw scenario at the given time step, planning problem and ego vehicle
        draw_object(scenario, ax=axis, draw_params={'time_begin': shown_timesteps[i]})
        draw_object(planning_problem_set, ax=axis)
        draw_object(
            obj=scenario.obstacle_by_id(ego_id),
            ax=axis,
            draw_params={
                'time_begin': shown_timesteps[i],
                'facecolor': 'g',
                'edgecolor': 'g',
            },
        )

        # focus on the ego vehicle
        axis.set_xlim(min(x) - 10, max(x) + 10)
        axis.set_ylim(min(y) - 5, max(y) + 5)

        # set title of the subplot
        axis.set(
            title='simulation time = ' + str(shown_timesteps[i] / 10) + ' s',
            xlabel=r'$x$ in m',
            ylabel=r'$y$ in m',
        )

        axis.set_aspect('equal')

    plt.show()


def show_solution_trajectory_with_gt_predictions(
    fontsize: float = 15.0,
    mode: int = 0,
    result: str = 'success',
    scenario_name: str = 'DEU_Ffb-1_6_T-1.xml',
    timestep: int = 0,
):
    """
    Create a figure that shows the vehicle at the given time step and highlights the trajectory.

    Scenario:
        mode=0, result='success', scenario_name='DEU_Ffb-1_6_T-1.xml' <------ Prediction problem

    Args:
        fontsize (float): Fontsize of the figure. Defaults to 15.
        mode (int): Mode used to solve the scenario. Defaults to 0.
        result (str): 'failure' or 'success'. Defaults to 'success'.
        scenario_name (str): Name of the scenario to be visualized. Defaults to 'DEU_Ffb-1_6_T-1.xml'.
        timestep (int): Timestep that should be shown. Defaults to 0.
    """
    # create the path of the file
    file_path = os.path.join(
        os.getcwd(),
        'planner/Frenet/results/solution trajectories/mode' + str(mode),
        result,
        scenario_name,
    )

    # set the fontsize of the figure
    plt.rcParams.update({'font.size': fontsize})

    # read the scenario, planning problem and the ID of the ego vehicle
    crfr = CommonRoadFileReader(file_path)
    scenario, planning_problem_set = crfr.open()
    ego_id = int(crfr._get_author())

    # get the trajectory of the ego vehicle
    ego_obst = scenario.obstacle_by_id(ego_id)
    ego_traj = ego_obst.prediction.trajectory.state_list
    x = [state.position[0] for state in ego_traj]
    y = [state.position[1] for state in ego_traj]

    # create the figure
    plt.figure(constrained_layout=False, figsize=(12, 12))

    # draw the planning problem
    draw_object(scenario.lanelet_network)

    # draw the obstacles
    for obstacle in scenario.obstacles:
        if obstacle.obstacle_id != ego_id:
            draw_object(obstacle, draw_params={'time_begin': timestep})
    # draw the ego obstacle
    draw_object(
        obj=scenario.obstacle_by_id(ego_id),
        draw_params={
            'time_begin': timestep,
            'facecolor': 'g',
            'edgecolor': 'g',
            'dynamic_obstacle': {'trajectory': {'draw_trajectory': False}},
        },
    )
    # focus on the ego vehicle
    plt.xlim(min(x) - 10, max(x) + 15)
    plt.ylim(min(y) - 10, max(y) + 30)

    plt.xlabel(r'$x$ in m')
    plt.ylabel(r'$y$ in m')
    plt.gca().set_aspect('equal')

    plt.show()


def show_solution_trajectory_and_course_of_the_states(
    fontsize: float = 15.0,
    mode: int = 0,
    result: str = 'failure',
    scenario_name: str = 'ZAM_Merge-1_1_T-1.xml',
):
    """
    Create a figure that shows the trajectory of the ego vehicle and the course of its states.

    Scenarios:
        mode=0, result='failure', scenario_name='ZAM_Merge-1_1_T-1.xml' <------- orientation failure
        mode=0, result='failure', scenario_name='USA_US101-28_3_T-1.xml' <------- ending lanelet failure

    Args:
        fontsize (float): Fontsize of the figure. Defaults to 15.
        mode (int): Mode used to solve the scenario. Defaults to 0.
        result (str): 'failure' or 'success'. Defaults to 'failure'.
        scenario_name (str): Name of the scenario to be visualized. Defaults to 'ZAM_Merge-1_1_T-1.xml'.
    """
    # create path of the file
    file_path = os.path.join(
        os.getcwd(),
        'planner/Frenet/results/solution trajectories/mode' + str(mode),
        result,
        scenario_name,
    )

    # set fontsize of the figure
    plt.rcParams.update({'font.size': fontsize})

    # get scenario, planning_problem and ID of the ego vehicle
    crfr = CommonRoadFileReader(file_path)
    scenario, planning_problem_set = crfr.open()
    planning_problem = planning_problem_set.planning_problem_dict[
        list(planning_problem_set.planning_problem_dict.keys())[0]
    ]
    ego_id = int(crfr._get_author())

    ego_obst = scenario.obstacle_by_id(ego_id)
    ego_traj = ego_obst.prediction.trajectory.state_list
    x = [state.position[0] for state in ego_traj]
    y = [state.position[1] for state in ego_traj]
    v = [state.velocity for state in ego_traj]
    t = [state.time_step / 10 for state in ego_traj]
    orientation = [state.orientation for state in ego_traj]

    fig = plt.figure(constrained_layout=False, figsize=(12, 12))
    gs = fig.add_gridspec(6, 7, left=0.1, right=0.95, wspace=0.04)
    ax1 = fig.add_subplot(gs[0:4, :])
    ax2 = fig.add_subplot(gs[5:, 0:3])
    ax3 = fig.add_subplot(gs[5, 4:])

    # show the planning problem
    draw_object(planning_problem_set, ax=ax1)

    # show final time step
    draw_object(obj=scenario, ax=ax1, draw_params={'time_begin': len(t) - 1})
    draw_object(
        obj=scenario.obstacle_by_id(ego_id),
        ax=ax1,
        draw_params={'time_begin': (len(t) - 1), 'facecolor': 'g', 'edgecolor': 'g'},
    )

    # show trajectory of the ego obstacle
    draw_object(scenario.obstacle_by_id(ego_id).prediction.trajectory, ax=ax1)

    # focus on the ego vehicle
    ax1.set_xlim(min(x) - 10, max(x) + 10)
    ax1.set_ylim(min(y) - 10, max(y) + 10)

    ax1.set(xlabel=r'$x$ in m', ylabel=r'$y$ in m')
    ax1.set_aspect('equal')

    # create velocity subplot
    ax2.plot(t, v, color='g')
    ax2.set(title='velocity', xlabel=r'$t$ in s', ylabel=r'$v$ in m/s')
    if hasattr(planning_problem.goal.state_list[0], 'velocity'):
        start = planning_problem.goal.state_list[0].velocity.start
        end = planning_problem.goal.state_list[0].velocity.end
        ax2.fill(
            [min(t), max(t), max(t), min(t)],
            [start, start, end, end],
            color=TUMColors.tum_orange.value,
            hatch='/',
            alpha=0.5,
            edgecolor=TUMColors.tum_orange.value,
            label='goal velocity',
        )
        ax2.legend()
    ax2.set_xlim(min(t), max(t))

    # create orientation subplot
    ax3.plot(t, orientation, color='g')
    ax3.set(title='orientation', xlabel=r'$t$ in s', ylabel=r'$\psi$ in rad')
    if hasattr(planning_problem.goal.state_list[0], 'orientation'):
        start = planning_problem.goal.state_list[0].orientation.start
        end = planning_problem.goal.state_list[0].orientation.end
        ax3.fill(
            [min(t), max(t), max(t), min(t)],
            [start, start, end, end],
            color=TUMColors.tum_orange.value,
            hatch='/',
            alpha=0.5,
            edgecolor=TUMColors.tum_orange.value,
            label='goal orientation',
        )
        ax3.legend()
    ax3.set_xlim(min(t), max(t))

    plt.show()


def show_scenario(
    fontsize: float = 15.0,
    mode: int = 0,
    result: str = 'failure',
    scenario_name: str = 'CHN_Sha-10_1_T-1.xml',
    timestep: int = 4,
):
    """
    Show the scenario for the given timestep and highlight the ego vehicle.

    Scenario:
        mode=0, result='failure', scenario_name='CHN_Sha-10_1_T-1.xml' <------- Failure obstacle cuts corner

    Args:
        fontsize (float): Fontsize of the figure. Defaults to 15.
        mode (int): Mode used to solve the scenario. Defaults to 0.
        result (str): 'failure' or 'success'. Defaults to 'failure'.
        scenario_name (str): Name of the scenario to be visualized. Defaults to 'CHN_Sha-10_1_T-1.xml'.
        timestep (int): Time step that should be shown. Defaults to 4.
    """
    # create path of the file
    file_path = os.path.join(
        os.getcwd(),
        'planner/Frenet/results/solution trajectories/mode' + str(mode),
        result,
        scenario_name,
    )

    # set fontsize of the figure
    plt.rcParams.update({'font.size': fontsize})

    # get scenario, planning problem and ID of the ego vehicle
    crfr = CommonRoadFileReader(file_path)
    scenario, planning_problem_set = crfr.open()
    ego_id = int(crfr._get_author())

    # create the figure
    plt.figure(constrained_layout=False, figsize=(12, 12))

    # draw the scenario and the planning problem
    draw_object(
        obj=scenario,
        draw_params={
            'time_begin': timestep,
            'dynamic_obstacle': {'trajectory': {'draw_trajectory': False}},
        },
    )
    draw_object(planning_problem_set)

    # draw the ego vehicle
    draw_object(
        obj=scenario.obstacle_by_id(ego_id),
        draw_params={
            'time_begin': timestep,
            'facecolor': 'g',
            'edgecolor': 'g',
            'dynamic_obstacle': {'trajectory': {'draw_trajectory': False}},
        },
    )

    plt.gca().set(xlabel=r'$x$ in m', ylabel=r'$y$ in m')
    plt.gca().set_aspect('equal')

    plt.show()


def compare_trajectories(
    fontsize: float = 15.0,
    mode: int = 2,
    result: str = 'success',
    scenario_names: [str] = ['global_path1.xml', 'ego_risk1.xml', 'max_risk1.xml'],
):
    """
    Create a figure that compares trajectories obtained with different cost functions.

    Args:
        fontsize (float): Fontsize of the figure. Defaults to 15.
        mode (int): Mode used to solve the scenario. Defaults to 2.
        result (str): 'failure' or 'success'. Defaults to 'success'.
        scenario_names [str]: Names of the xml files that should be compared. Defaults to ['global_path1.xml', 'ego_risk1.xml', 'max_risk1.xml'].
    """
    # get file paths
    file_path1 = os.path.join(
        os.getcwd(),
        'planner/Frenet/results/solution trajectories/mode' + str(mode),
        result,
        scenario_names[0],
    )
    file_path2 = os.path.join(
        os.getcwd(),
        'planner/Frenet/results/solution trajectories/mode' + str(mode),
        result,
        scenario_names[1],
    )
    file_path3 = os.path.join(
        os.getcwd(),
        'planner/Frenet/results/solution trajectories/mode' + str(mode),
        result,
        scenario_names[2],
    )

    # set fontsize of the figure
    plt.rcParams.update({'font.size': fontsize})

    # read sceanrio 1
    crfr1 = CommonRoadFileReader(file_path1)
    scenario1, planning_problem_set = crfr1.open()

    # read sceanrio 2
    crfr2 = CommonRoadFileReader(file_path2)
    scenario2, planning_problem_set = crfr2.open()

    # read sceanrio 3
    crfr3 = CommonRoadFileReader(file_path3)
    scenario3, planning_problem_set = crfr3.open()

    # get ID of the ego vehicle
    ego_id = int(crfr1._get_author())

    # get trajectories
    ego_obst1 = scenario1.obstacle_by_id(ego_id)
    ego_traj1 = ego_obst1.prediction.trajectory.state_list
    x1 = [state.position[0] for state in ego_traj1]
    y1 = [state.position[1] for state in ego_traj1]

    ego_obst2 = scenario2.obstacle_by_id(ego_id)
    ego_traj2 = ego_obst2.prediction.trajectory.state_list
    x2 = [state.position[0] for state in ego_traj2]
    y2 = [state.position[1] for state in ego_traj2]

    ego_obst3 = scenario3.obstacle_by_id(ego_id)
    ego_traj3 = ego_obst3.prediction.trajectory.state_list
    x3 = [state.position[0] for state in ego_traj3]
    y3 = [state.position[1] for state in ego_traj3]

    # create figure
    plt.figure(figsize=(20, 7))

    # draw lanelet network and obstacles
    draw_object(obj=scenario1.lanelet_network)
    draw_object(
        obj=ego_obst1,
        draw_params={
            'time_begin': 0,
            'facecolor': 'g',
            'edgecolor': 'g',
            'dynamic_obstacle': {'trajectory': {'draw_trajectory': False}},
        },
    )
    for obstacle in scenario1.obstacles:
        if obstacle.obstacle_id != ego_id:
            draw_object(obstacle)

    # plot the trajectories
    plt.plot(
        x1,
        y1,
        zorder=26,
        color=TUMColors.tum_gray.value,
        linestyle='--',
        label='no risk considered',
        linewidth=3,
    )
    plt.plot(
        x3,
        y3,
        zorder=26,
        color=TUMColors.tum_orange.value,
        linestyle='-.',
        label='minimized maximum risk',
        linewidth=3,
    )
    plt.plot(
        x2,
        y2,
        zorder=26,
        color=TUMColors.tum_blue.value,
        linestyle=':',
        label='minimized ego risk',
        linewidth=3,
    )

    # focus on the ego vehicle
    plt.gca().set_xlim(min(x1) - 3, max(x1) - 10)
    plt.gca().set_ylim(min(y1) - 6, max(y1) + 8)

    plt.xlabel(r'$x$ in m')
    plt.ylabel(r'$y$ in m')
    plt.legend(loc=1).set_zorder(25)
    plt.gca().set_aspect('equal')
    plt.subplots_adjust(left=0.07, right=0.99)

    plt.show()


def create_custom_scenario():
    """Create a custom scenario with a bicycle on the right side and a car on the left lane."""
    # get the initial scenario that gets changed
    scenario_name = 'hand-crafted/ZAM_Zip-1_29_T-1.xml'

    file_path = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        'commonroad-scenarios/scenarios',
        scenario_name,
    )

    crfr = CommonRoadFileReader(file_path)
    scenario, _ = crfr.open()

    # create the planning problem
    initial_state = State(
        position=[-170, 5.2393],
        orientation=0.0,
        velocity=15.0,
        yaw_rate=0.0,
        slip_angle=0.0,
        time_step=0,
    )
    goal_state = State(time_step=Interval(start=40, end=41))
    goal_region = GoalRegion(state_list=[goal_state])
    new_planning_problem = PlanningProblem(
        planning_problem_id=999, initial_state=initial_state, goal_region=goal_region
    )

    # remove all obstacles from the initial scenario
    for obstacle in scenario.obstacles:
        scenario.remove_obstacle(obstacle)

    # create new obstacle
    # car on the left lane
    obstacle_id = 196
    obstacle_shape = Rectangle(length=4.5, width=2)
    obstacle_type = ObstacleType.CAR
    obst_init_state = State(
        position=np.array([-170, 8.72]),
        orientation=0.0,
        velocity=15.0,
        yaw_rate=0.0,
        slip_angle=0.0,
        time_step=0,
    )

    # create the state list of the obstacle
    # the car drives straight forward with a velocity of 15 m/s
    obst_state_list = []
    for i in range(1, 85):
        new_position = np.array(
            [obst_init_state.position[0] + 0.1 * 15 * i, obst_init_state.position[1]]
        )
        new_state = State(
            position=new_position, velocity=15, orientation=0.0, time_step=i
        )
        obst_state_list.append(new_state)

    # create the trajectory and the prediction of the car
    obst_traj = Trajectory(initial_time_step=1, state_list=obst_state_list)
    obst_pred = TrajectoryPrediction(obst_traj, obstacle_shape)

    # initialize the dynamic obstacle representing the car
    obst = DynamicObstacle(
        obstacle_id=obstacle_id,
        obstacle_shape=obstacle_shape,
        obstacle_type=obstacle_type,
        initial_state=obst_init_state,
        prediction=obst_pred,
    )
    # add the obstacle to the scenario
    scenario.add_objects(obst)

    # create a bicycle on the right side of the road
    obstacle_id = 197
    obstacle_shape = Rectangle(length=1.5, width=0.7)
    obstacle_type = ObstacleType.BICYCLE
    # initial state of the bicycle
    obst_init_state = State(
        position=np.array([-160, 4]),
        orientation=0.0,
        velocity=5.0,
        yaw_rate=0.0,
        slip_angle=0.0,
        time_step=0,
    )

    # create the state list of the bicycle
    # the bicycle drives in a straight line with a velocity of 5 m/s
    obst_state_list = []
    for i in range(1, 85):
        new_position = np.array(
            [obst_init_state.position[0] + 0.1 * 5.0 * i, obst_init_state.position[1]]
        )
        new_state = State(
            position=new_position, velocity=5.0, orientation=0.0, time_step=i
        )
        obst_state_list.append(new_state)

    # create the trajectory and prediction of the bicycle
    obst_traj = Trajectory(initial_time_step=1, state_list=obst_state_list)
    obst_pred = TrajectoryPrediction(obst_traj, obstacle_shape)

    # create the dynamic obstacle representing the bicycle
    obst = DynamicObstacle(
        obstacle_id=obstacle_id,
        obstacle_shape=obstacle_shape,
        obstacle_type=obstacle_type,
        initial_state=obst_init_state,
        prediction=obst_pred,
    )

    # add the bicycle to the scenario
    scenario.add_objects(obst)

    # show the scenario and the planning problem
    draw_object(scenario)
    draw_object(new_planning_problem)

    plt.autoscale()
    plt.gca().set_aspect('equal')
    plt.show()

    # save the created scenario
    author = 'Florian Pfab'
    affiliation = 'Technical University of Munich, Germany'
    source = ''

    # write new scenario and add the created planning problem
    planning_problem_set = PlanningProblemSet(
        planning_problem_list=[new_planning_problem]
    )
    fw = CommonRoadFileWriter(
        scenario, planning_problem_set, author, affiliation, source
    )

    filename = "test_scenario.xml"
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)


if __name__ == '__main__':
    # create_bar_chart_and_tables(fontsize=28, batch='0-1000mode0', only_show_one='calculate costs')
    # show_solution_trajectory_in_one_plot(fontsize=15, n_shown_states=5, mode=0, result='failure', scenario_name='ZAM_Merge-1_1_T-1.xml')
    # show_solution_trajectory_in_four_plots(fontsize=20, mode=0, result='success', scenario_name='RUS_Bicycle-4_1_T-1.xml')
    # show_solution_trajectory_and_course_of_the_states(fontsize=20, mode=0, result='failure', scenario_name='ZAM_Merge-1_1_T-1.xml')
    # show_scenario(fontsize=20, mode=0, result='failure', scenario_name='CHN_Sha-10_1_T-1.xml')
    # show_solution_trajectory_with_gt_predictions(fontsize=20, mode=0, result='success', scenario_name='DEU_Ffb-1_6_T-1.xml', timestep=0)
    # compare_trajectories(fontsize=20, mode=2, result='success', scenario_names=['global_path1.xml', 'ego_risk1.xml', 'max_risk1.xml'])
    create_custom_scenario()
