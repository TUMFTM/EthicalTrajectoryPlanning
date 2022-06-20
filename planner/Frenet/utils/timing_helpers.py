#!/user/bin/env python

"""Helper functions to get the execution times of the frenét planner."""

import time
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def add_exec_time(exec_times: dict, func_name: str, time: float):
    """
    Add an execution time to the execution times dictionary.

    Args:
        exec_times (dict): Dictionary with the execution times.
        func_name (str): Name of the timed function.
        time (float): Execution time.

    Returns:
        dict: Dictionary with the execution times
    """
    # check if the name of the function is already available in the dictionary
    if func_name in exec_times.keys():
        exec_times[func_name].append(time)
    # otherwise create a new entry
    else:
        exec_times[func_name] = [time]

    return exec_times


def create_pie_and_table_plot(
    key: str,
    sub_keys: [str],
    exec_times_dict: dict,
    col_labels: [str],
    fontsize: int = 12,
):
    """
    Create a figure with a pie chart and a table.

    Args:
        key (str): Name of the main function.
        sub_keys ([str]): Name of the sub functions.
        exec_times_dict (dict): Dictionary with the execution times.
        col_labels ([str]): Headers of the columns.
        fontsize (int): Fontsize. Defaults to 12.

    Returns:
        fig: Figure of the pie chart and the table.
    """
    # create the figure with 2 subplots
    fig = plt.figure(constrained_layout=True, figsize=(20, 12))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # get the data for the pie chart and the table
    pie_total_times, pie_keys, table_text = get_plotting_data(
        general_key=key, sub_keys=sub_keys, exec_times_dict=exec_times_dict
    )

    # create the pie chart
    ax1.pie(x=pie_total_times, labels=pie_keys, autopct="%1.1f%%", startangle=90)
    ax1.set(title=key)
    ax1.axis("equal")

    # create the table
    table = ax2.table(cellText=table_text, loc="center", colLabels=col_labels)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    ax2.axis("off")

    return fig


def create_str_for_exec_times(exec_times_dict: dict, col_labels: [str]):
    """
    Create a string containing all important data from the execution time dictionary.

    Args:
        exec_times_dict (dict): Dictionary with the execution times.
        col_labels ([str]): Labels for the columns.

    Returns:
        str: All important data from the execution times dictionary.
    """
    return_str = ""
    # add the labels of the columns
    for col_label in col_labels:
        return_str += col_label
        return_str += ";"
    return_str += "\n"
    # add the information about the execution time
    for key in exec_times_dict.keys():
        return_str += key + ";"
        return_str += str(round(np.sum(exec_times_dict[key]), 10)) + ";"
        return_str += str(len(exec_times_dict[key])) + ";"
        return_str += str(round(np.mean(exec_times_dict[key]), 10)) + "\n"

    return return_str


def plot_exec_time_charts(
    exec_times_dict: dict,
    save: bool = True,
    save_animation: bool = False,
    show: bool = False,
):
    """
    Create figures that give information about the execution times.

    Args:
        exec_times_dict (dict): Dictionary with the execution times.
        save (bool): True if the figures should be saved. Defaults to True.
        save_animation (bool): True if information about the animation process should be saved. Defaults to False.
    """
    plt.close()
    # get the labels of the columns
    col_labels = ["Function", "total time (s)", "number of executions", "avg time (s)"]
    # get the names of the processes and their subprocesses
    processes_and_subprocesses = get_processes_and_subprocesses(
        with_animation=save_animation
    )

    # create a txt file that holds all the information about the execution times
    if save:
        # get the path of the file and create a folder that holds the exact time of the execution
        file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        Path(file_path + "/results/execution times/" + timestr).mkdir(
            parents=True, exist_ok=True
        )
        # create the log file
        log_file = open(
            os.path.join(
                file_path,
                ("./results/execution times/" + timestr + "/execution_times.txt"),
            ),
            "w",
        )
        log_file.write(
            create_str_for_exec_times(
                exec_times_dict=exec_times_dict, col_labels=col_labels
            )
        )
        log_file.close()

    # create and save a figure with a pie chart and a table for every process
    for key in processes_and_subprocesses.keys():
        fig = create_pie_and_table_plot(
            key=key,
            sub_keys=processes_and_subprocesses[key],
            exec_times_dict=exec_times_dict,
            col_labels=col_labels,
        )
        if save:
            plt.savefig(
                fname=(file_path + "/results/execution times/" + timestr + "/" + key),
                format="pdf",
            )
        elif show:
            plt.show(fig)


def get_plotting_data(general_key: str, sub_keys: [str], exec_times_dict: dict):
    """
    Get the data from the execution times dictionary.

    Args:
        general_key (str): Name of the general function.
        sub_keys ([str]): Names of the subfunctions.
        exec_times_dict (dict): Dictionary with the execution times.

    Returns:
        [float]: Total times shown in the pie chart.
        [str]: Keys of the times shown in the pie chart.
        [str]: Data for the table.
    """
    # get the total time of the porcess
    total_time = np.sum(exec_times_dict[general_key])

    keys = sub_keys
    total_times = []
    avg_times = []
    n_execs = []

    # get the information about the subprocesses
    for key in keys:
        total_times.append(np.sum(exec_times_dict[key]))
        avg_times.append(np.mean(exec_times_dict[key]))
        n_execs.append(len(exec_times_dict[key]))

    # get the times that are not captured by subprocesses
    keys.append("other stuff")
    total_times.append(max(0.0, (total_time - np.sum(total_times))))
    n_execs.append(len(exec_times_dict[general_key]))
    avg_times.append(total_times[-1] / n_execs[-1])

    # create the a matrix that represents the table with all the information
    table = []
    for i in range(len(keys)):
        row = [keys[i], round(total_times[i], 5), n_execs[i], round(avg_times[i], 5)]
        table.append(row)
    table.append(
        [
            "total time",
            round(total_time, 5),
            len(exec_times_dict[general_key]),
            round(total_time / len(exec_times_dict[general_key]), 5),
        ]
    )

    # get the data necessary for the pie chart
    pie_total_times = []
    pie_keys = []
    total_time_small_stuff = 0.0

    # processes that take less than 1 % of the execution time are put together as 'other stuff' to improve readability
    for i in range(len(keys)):
        if total_times[i] < total_time / 100.0 or keys[i] == "other stuff":
            total_time_small_stuff += total_times[i]
        else:
            pie_total_times.append(total_times[i])
            pie_keys.append(keys[i])

    pie_keys.append("other stuff")
    pie_total_times.append(total_time_small_stuff)

    return pie_total_times, pie_keys, table


def evaluate_exec_times(exec_times: dict):
    """
    Print some basic information about the execution time of the evaluation.

    Args:
        exec_times (dict): Dictionary with the execution times.
    """
    for key in exec_times.keys():
        n_executions = len(exec_times[key])
        avg_exec_time = np.mean(exec_times[key])
        print(
            key
            + ":\tnumber of executions: "
            + str(n_executions)
            + "\taverage execution time: "
            + str(round(avg_exec_time, 5))
            + " s\ttotal execution time: "
            + str(round(np.sum(exec_times[key]), 5))
            + " s"
        )


def merge_dicts(exec_times_dicts):
    """
    Merge multiple dictionaries.

    Args:
        exec_times_dicts ([dict]): The dicts that should be merged.

    Returns:
        dict: The merged dict.
    """
    full_dict = {}
    for dict_item in exec_times_dicts:
        for key in dict_item.keys():
            for value in dict_item[key]:
                full_dict = add_exec_time(
                    exec_times=full_dict, func_name=key, time=value
                )

    return full_dict


def get_processes_and_subprocesses(with_animation: bool = False):
    """
    Return a dictionary that holds a list of all processes and subprocesses as used for the timing.

    Args:
        with_animation (bool): True if the animation is considered when timing. Defaults to False.

    Returns:
        dict: Dictionary that holds all processes and subprocesses used for the timing.
    """
    processes_and_subprocesses = {
        "total time": [
            "read scenario",
            "add vehicle to scenario",
            "read vehicle parameters",
            "initialization",
            "simulation",
            "plot trajectories",
        ],
        "initialization": [
            "initialize planner",
            "plan global path",
            "check curvature of path",
            "initialize road boundary",
            "initialize collision checker",
            "initialize goal area",
        ],
        "simulation": [
            "update driven trajectory",
            "problem solved?",
            "get v list",
            "calculate trajectories",
            "prediction",
            "sort trajectories",
        ],
        "calculate trajectories": [
            "initialize quartic polynomial",
            "calculate quartic polynomial",
            "initialize quintic polynomial",
            "calculate quintic polynomial",
            "calculate global trajectory",
            "initialize trajectory",
        ],
        "calculate global trajectory": [
            "convert to ds/dt",
            "calculate reference points",
            "calculate reference gradients",
            "calculate reference yaw",
            "calculate reference curvature",
            "calculate reference curvature derivation",
            "calculate trajectory states",
        ],
        "sort trajectories": [
            "check validity",
            "calculate costs",
            "sort list by costs",
        ],
        "check validity": [
            "check velocity",
            "check acceleration",
            "check curvature",
            "check road boundaries",
            "check collision",
        ],
        "calculate costs": [
            "calculate risk",
            "calculate visible area",
            "calculate jerk",
            "calculate velocity",
            "calculate distance to global path",
            "calculate travelled distance",
            "calculate distance to goal pos",
            "calculate distance to lane center",
            "multiply weights and costs",
            "get cost factor",
        ],
        "get cost factor": [
            "goal reached?",
            "goal reached in time?",
            "goal reached and left?",
        ],
    }

    if with_animation is False:
        return processes_and_subprocesses
    else:
        processes_and_subprocesses["total time"].append("animate scenario")
        processes_and_subprocesses["animate scenario"] = [
            "animate save",
            "animate create animation",
            "animate create states",
        ]
        return processes_and_subprocesses


# EOF
