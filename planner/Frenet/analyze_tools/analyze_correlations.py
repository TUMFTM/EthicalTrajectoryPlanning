"""Analyze trajectory costs for correlations."""

import os
from analyze_log import FrenetLogVisualizer
from itertools import groupby
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from math import isnan
import multiprocessing
import progressbar
import json
import argparse
import traceback


def all_equal(iterable):
    """Check if all are equal.

    Args:
        iterable ([type]): [description]

    Returns:
        [type]: [description]
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


corr_mat_list = []
long_mat_list = []
lat_mat_list = []
dist_mat_list = []
corr_dict = {}
long_dict = {}
lat_dict = {}
dist_dict = {}
scenario_list = []
key_list = []

cpu_count = 60  # multiprocessing.cpu_count()


def eval_func(logfile):
    """Calculate correlations for a logfile.

    Args:
        logfile ([type]): [description]

    Returns:
        [type]: [description]
    """
    try:
        frenet_log = FrenetLogVisualizer(
            logfile, visualize=False, verbose=False
        )
        corr_mat, keys = frenet_log.correlation_matrix(plot=False)
        long_mat, lat_mat, dist_mat = frenet_log.distance_matrix(plot=False)

        return corr_mat, long_mat, lat_mat, dist_mat, keys, logfile
    except Exception:
        traceback.print_exc()
        return None


def process_return_dict(return_list):
    """Unpack return list from multiprocessing.

    Args:
        return_list ([type]): [description]
    """
    if return_list is not None:
        # Filter nans
        if (return_list[0] == return_list[0]).all():
            corr_mat_list.append(return_list[0])
            long_mat_list.append(return_list[1])
            lat_mat_list.append(return_list[2])
            dist_mat_list.append(return_list[3])
            key_list.append(return_list[4])
            scenario_list.append(return_list[5].split("/")[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./planner/Frenet/results/logs")
    args = parser.parse_args()

    log_file_list = os.listdir(args.logdir)
    log_file_list = [os.path.join(args.logdir, l) for l in log_file_list]
    result_dir = os.path.dirname(args.logdir)

    with progressbar.ProgressBar(max_value=len(log_file_list)).start() as pbar:
        with multiprocessing.Pool(processes=cpu_count) as pool:
            for return_list in pool.imap_unordered(eval_func, log_file_list):
                process_return_dict(return_list)
                pbar.update(pbar.value + 1)

    if not all_equal(key_list):
        print(
            "Error: Keys are ambiguous, but must be the same. Make sure to run with the same settings."
        )
        longest_len = max([len(i) for i in key_list])
        new_key_list = []
        new_scenario_list = []
        new_corr_mat_list = []
        for key, scen, corr in zip(key_list, scenario_list, corr_mat_list):
            if len(key) == longest_len:
                new_key_list.append(key)
                new_scenario_list.append(scen)
                new_corr_mat_list.append(corr)
            else:
                print(f"I had to remove {scen} with keys {key}")

        print(key_list)

    for i in range(len(key_list[0])):
        for j in range(i + 1, len(key_list[0])):
            corr_dict[str(key_list[0][i]) + "<->" + str(key_list[0][j])] = [
                corr_mat[i, j] for corr_mat in corr_mat_list
            ]
            long_dict[str(key_list[0][i]) + "<->" + str(key_list[0][j])] = [
                long_mat[i, j] for long_mat in long_mat_list
            ]
            lat_dict[str(key_list[0][i]) + "<->" + str(key_list[0][j])] = [
                lat_mat[i, j] for lat_mat in lat_mat_list
            ]
            dist_dict[str(key_list[0][i]) + "<->" + str(key_list[0][j])] = [
                dist_mat[i, j] for dist_mat in dist_mat_list
            ]

    corr_dict_scenes = {}
    long_dict_scenes = {}
    lat_dict_scenes = {}
    dist_dict_scenes = {}

    for key in corr_dict:
        corr_dict_scenes[key] = {}
        long_dict_scenes[key] = {}
        lat_dict_scenes[key] = {}
        dist_dict_scenes[key] = {}

        for idx, val in enumerate(corr_dict[key]):
            corr_dict_scenes[key][scenario_list[idx]] = corr_dict[key][idx]
            long_dict_scenes[key][scenario_list[idx]] = long_dict[key][idx]
            lat_dict_scenes[key][scenario_list[idx]] = lat_dict[key][idx]
            dist_dict_scenes[key][scenario_list[idx]] = dist_dict[key][idx]

        # corr_dict_scenes[key].sort()
        corr_dict_scenes[key] = dict(sorted(corr_dict_scenes[key].items(), key=lambda item: item[1]))
        long_dict_scenes[key] = dict(sorted(long_dict_scenes[key].items(), key=lambda item: item[1], reverse=True))
        lat_dict_scenes[key] = dict(sorted(lat_dict_scenes[key].items(), key=lambda item: item[1], reverse=True))
        dist_dict_scenes[key] = dict(sorted(dist_dict_scenes[key].items(), key=lambda item: item[1], reverse=True))

    if len(corr_mat_list) != len(scenario_list):
        print(f"Warning: Scenario list ({len(scenario_list)}) has not the same lentgh as corr_dict ({len(corr_mat_list)})")

    with open(os.path.join(result_dir, "corr_dict.json"), "w") as fp:
        json.dump(corr_dict, fp)

    with open(os.path.join(result_dir, "long_dict.json"), "w") as fp:
        json.dump(long_dict, fp)

    with open(os.path.join(result_dir, "lat_dict.json"), "w") as fp:
        json.dump(lat_dict, fp)

    with open(os.path.join(result_dir, "dist_dict.json"), "w") as fp:
        json.dump(dist_dict, fp)

    with open(os.path.join(result_dir, "scen_list.txt"), "w") as fp:
        json.dump(scenario_list, fp)

    with open(os.path.join(result_dir, "corr_dict_scenes.json"), "w") as fp:
        json.dump(corr_dict_scenes, fp)

    with open(os.path.join(result_dir, "long_dict_scenes.json"), "w") as fp:
        json.dump(long_dict_scenes, fp)

    with open(os.path.join(result_dir, "lat_dict_scenes.json"), "w") as fp:
        json.dump(lat_dict_scenes, fp)

    with open(os.path.join(result_dir, "dist_dict_scenes.json"), "w") as fp:
        json.dump(dist_dict_scenes, fp)

    clean_corr_dict = {k: corr_dict[k] for k in corr_dict if not isnan(sum(corr_dict[k]))}

    fig, ax = plt.subplots()
    ax.boxplot(clean_corr_dict.values())
    ax.set_xticklabels(clean_corr_dict.keys(), rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "correlations.pdf"))

    print("Done.")
