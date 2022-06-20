#!/user/bin/env python

"""Evaluation script for the global path planner."""

# Standard imports
import os
import sys
import time
import random
import matplotlib.pyplot as plt

# 3rd party imports
import progressbar
import argparse
from PyPDF2 import PdfFileMerger
from joblib import Parallel, delayed
import multiprocessing

from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.common.file_reader import CommonRoadFileReader

module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(module_path)

from planner.GlobalPath.lanelet_based_planner import LaneletPathPlanner

__author__ = "Florian Pfab"
__email__ = "Florian.Pfab@tum.de"
__date__ = "23.05.2020"


def evaluate_scenario(filename):
    """
    Evaluate a given scenario.

    Args:
        filename (str): Filename of the scenarios to be evaluated.

    Returns:
        array: Containing alternating a bool if the goal was reached and a log msg with relevant facts about the global path planning process. Has 2 * number of planning problems for the scenario entries.

    """
    planning_time = 0.0
    plotting_time = 0.0
    load_scenarios_time = 0.0

    ev_time0 = time.time()

    # Read scenario
    scenario, planning_problem_set = CommonRoadFileReader(filename).open()

    load_scenarios_time += time.time() - ev_time0

    # Create return array
    return_array = []

    # Get planning problem
    for planning_problem_id in range(len(list(planning_problem_set.planning_problem_dict.values()))):
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[planning_problem_id]

        # Try to find the global path
        try:
            # Initialize motion planner
            path_planner = LaneletPathPlanner(scenario, planning_problem)

            start_time = time.time()
            # Execute the search
            path, path_length = path_planner.plan_global_path()

            planning_time = time.time() - start_time

            if path is not None:
                goal_reached = True
            else:
                goal_reached = False

            log_msg = scenario.benchmark_id + '; ' + str(planning_problem.planning_problem_id) + '; ' + str(goal_reached) + '; ' + str(round(planning_time, 5)) + '; ' + str(round(path_length, 2)) + '\n'

        except Exception:
            goal_reached = False
            log_msg = scenario.benchmark_id + '; ' + '; FAILED with Error: ' + str(sys.exc_info()[0]) + '; Error-message: ' + str(sys.exc_info()[1]) + '\n'

        ev_time1 = time.time()

        # Plot and save path
        plt.figure(figsize=[15, 10])

        try:
            plt.title(scenario.benchmark_id + ' ; Planning Problem ID: ' + str(planning_problem.planning_problem_id) + " Succeeded")
            plt.axis('equal')
            draw_object(scenario.lanelet_network)
            draw_object(planning_problem)
            path_to_goal, path_after_goal = path_planner.split_global_path(global_path=path)
            plt.plot(path_to_goal[:, 0], path_to_goal[:, 1], color='red', zorder=20)
            plt.plot(path_after_goal[:, 0], path_after_goal[:, 1], color='red', zorder=20, linestyle='--')
            plt.xlabel('x in m')
            plt.ylabel('y in m')
            plt.savefig('./results/path_plots/succeeded/' + scenario.benchmark_id + str('__') + str(planning_problem.planning_problem_id) + '.pdf')
            plt.close()

        except Exception:
            plt.title(scenario.benchmark_id + "; Planning Problem ID: " + str(planning_problem.planning_problem_id) + " Failed")
            plt.axis('equal')
            draw_object(scenario.lanelet_network)
            draw_object(planning_problem)
            plt.savefig('./results/path_plots/failed/' + scenario.benchmark_id + str('__') + str(planning_problem.planning_problem_id) + '.pdf')
            plt.close()

        plotting_time += time.time() - ev_time1

        # Create dict with times
        time_dict = {
            "planning": planning_time,
            "plotting": plotting_time,
            "load": load_scenarios_time,
            "total": (time.time() - ev_time0)
        }

        return_array.append(goal_reached)
        return_array.append(log_msg)
        return_array.append(time_dict)

    return return_array


if __name__ == '__main__':
    # Timing
    time0 = time.time()

    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_scenarios', type=bool, default=False)
    args = parser.parse_args()

    # Change the working directory to the directory of the evaluation script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Get the directory of the scenarios
    scenario_directory = "../../../commonroad-scenarios/scenarios/"

    filelist = []

    # Get all files in the directory and the subdirectories
    for root, dirs, files in os.walk(scenario_directory):
        for file in files:
            filelist.append(os.path.join(root, file))

    # If not all scenarios should be evaluated, 10 random ones are evaluated
    if args.all_scenarios is False:
        new_filelist = []
        for i in range(10):
            new_filelist.append(random.choice(filelist))
        filelist = new_filelist

    result_directory = './results/'
    path_plots_directory = './results/path_plots/'
    path_plots_directory_succeeded = './results/path_plots/succeeded/'
    path_plots_directory_failed = './results/path_plots/failed/'

    # Create directories if they dont exist
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    if not os.path.exists(path_plots_directory):
        os.makedirs(path_plots_directory)
    if not os.path.exists(path_plots_directory_succeeded):
        os.makedirs(path_plots_directory_succeeded)
    if not os.path.exists(path_plots_directory_failed):
        os.makedirs(path_plots_directory_failed)

    # Get number of available cores
    num_cores = multiprocessing.cpu_count()
    print("Running on {} cores".format(num_cores))

    time_pre_multi = time.time()

    # Evaluate scenarios on all cores
    results = Parallel(n_jobs=num_cores)(delayed(evaluate_scenario)(i) for i in progressbar.progressbar(filelist))

    # without multiprocessing
    # results = []
    # for file in filelist:
    #     results.append(evaluate_scenario(file))

    time_multi_main = time.time() - time_pre_multi

    goal_reached_list = []
    log_msgs = []
    n_scenarios = len(results)
    n_planning_problems = 0
    planning_time = 0.0
    load_scenarios_time = 0.0
    plotting_time = 0.0
    multi_time = 0.0

    for scenario_counter in range(len(results)):
        for planning_problem_counter in range(int(len(results[scenario_counter]) / 3)):
            n_planning_problems += 1
            goal_reached_list.append(results[scenario_counter][0 + planning_problem_counter * 3])
            log_msgs.append(results[scenario_counter][1 + planning_problem_counter * 3])
            time_dict = results[scenario_counter][2 + planning_problem_counter * 3]
            planning_time += time_dict["planning"]
            plotting_time += time_dict["plotting"]
            load_scenarios_time += time_dict["load"]
            multi_time += time_dict["total"]

    # Get number of succeeded/failed scenarios
    suc_count = goal_reached_list.count(True)
    fail_count = goal_reached_list.count(False)
    suc_rate = suc_count / n_planning_problems * 100

    # Create log file
    log_file = open(os.path.join(result_directory, 'result_logs.txt'), 'w')
    log_file.write("Benchmark ID; Planning Problem; Succeeded; Execution Time; Path Length\n")
    for msg in log_msgs:
        log_file.write(msg)
    log_file.write('\n%d scenarios containing %d planning problems were evaluated.\nSuccess-rate: %.2f %%' % (n_scenarios, n_planning_problems, suc_rate))
    log_file.close()

    print('Evaluation finished. %d scenarios containing %d planning problems were evaluated.\nSuccess-rate: %.2f %%' % (n_scenarios, n_planning_problems, suc_rate))

    time1 = time.time()

    # Merge PDFs
    merger = PdfFileMerger()
    for pdf in os.listdir(path_plots_directory_succeeded):
        merger.append(os.path.join(path_plots_directory_succeeded, pdf))

    merger.write(os.path.join(result_directory, 'summary_succeeded.pdf'))
    merger.close()

    merger = PdfFileMerger()
    for pdf in os.listdir(path_plots_directory_failed):
        merger.append(os.path.join(path_plots_directory_failed, pdf))

    merger.write(os.path.join(result_directory, 'summary_failed.pdf'))
    merger.close()

    merger = PdfFileMerger()
    for file in os.listdir(result_directory):
        if file.endswith(".pdf"):
            merger.append(os.path.join(result_directory, file))

    merger.write(os.path.join(result_directory, 'summary.pdf'))
    merger.close()

    create_pdfs_time = time.time() - time1
    total_time = time.time() - time0

    # Calculate the time percentages of the different tasks
    percentage_create_pdfs = create_pdfs_time / total_time
    percentage_multi = time_multi_main / total_time
    percentage_planning = percentage_multi * (planning_time / multi_time)
    percentage_plotting = percentage_multi * (plotting_time / multi_time)
    percentage_load_scenarios = percentage_multi * (load_scenarios_time / multi_time)
    percentage_other_stuff = 1 - (percentage_create_pdfs + percentage_planning + percentage_plotting + percentage_load_scenarios)

    # Get the average times for planning, plotting and loading
    av_planning_time = planning_time / n_planning_problems
    av_plotting_time = plotting_time / n_planning_problems
    av_load_time = load_scenarios_time / n_planning_problems

    # Plot the summary of the evaluation
    plt.subplot(221)
    plt.suptitle('Evaluation summary')
    plt.axis('equal')
    labels = 'Other stuff', 'Creating PDFs', 'Planning', 'Plotting', 'Loading Scenarios'
    sizes = [percentage_other_stuff, percentage_create_pdfs, percentage_planning, percentage_plotting, percentage_load_scenarios]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.subplot(222)
    plt.barh(['Success rate', 'Success rate'], [suc_rate, 100 - suc_rate], left=[0, suc_rate], color=['g', 'r'], height=15)
    plt.text(suc_rate / 2, 0, str(round(suc_rate, 2)) + ' %', ha='center', va='center')
    ax = plt.gca()
    ax.set_xlim(0, 100)
    plt.axis('equal')
    plt.subplot(212)
    plt.axis('off')
    table_text = [['Evaluated scenarios', n_scenarios],
                  ['Evaluated planning problems', n_planning_problems],
                  ['Failed planning problems', fail_count],
                  ['Solved planning problems', suc_count],
                  ['Success rate', str(round(suc_rate, 2)) + ' %'],
                  ['Evaluation Duration', str(round(total_time, 2)) + ' s'],
                  ['Average scenario loading time', str(round(av_load_time, 5)) + ' s'],
                  ['Average planning time', str(round(av_planning_time, 5)) + ' s'],
                  ['Average plotting time', str(round(av_plotting_time, 5)) + ' s']]
    plt.table(table_text, cellLoc='left', loc='center')
    plt.subplots_adjust(wspace=1.0)
    plt.savefig('./results/evaluation.pdf')
    plt.show()
