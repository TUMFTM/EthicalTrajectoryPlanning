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


def all_equal(iterable):
    """Check if all are equal.

    Args:
        iterable ([type]): [description]

    Returns:
        [type]: [description]
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


logdir = "./planner/Frenet/results/logs"

corr_mat_list = []
corr_dict = {}
scenario_list = []
key_list = []

cpu_count = 60  # multiprocessing.cpu_count()
log_file_list = os.listdir(logdir)


def eval_func(logfile):
    """Calculate correlations for a logfile.

    Args:
        logfile ([type]): [description]

    Returns:
        [type]: [description]
    """
    try:
        frenet_log = FrenetLogVisualizer(
            os.path.join(logdir, logfile), visualize=False, verbose=False
        )
        corr_mat, keys = frenet_log.correlation_matrix(plot=False)

        return corr_mat, keys, logfile
    except Exception:
        return None


def process_return_dict(return_list):
    """Unpack return list from multiprocessing.

    Args:
        return_list ([type]): [description]
    """
    if return_list is not None:
        corr_mat_list.append(return_list[0])
        key_list.append(return_list[1])
        scenario_list.append(return_list[2])


with progressbar.ProgressBar(max_value=len(log_file_list)).start() as pbar:
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for return_list in pool.imap_unordered(eval_func, log_file_list):
            process_return_dict(return_list)
            pbar.update(pbar.value + 1)


if not all_equal(key_list):
    print(
        "Error: Keys are ambiguous, but must be the same. Make sure to run with the same settings."
    )

for i in range(len(key_list[0])):
    for j in range(i + 1, len(key_list[0])):
        corr_dict[str(key_list[0][i]) + "<->" + str(key_list[0][j])] = [
            corr_mat[i, j] for corr_mat in corr_mat_list
        ]

for key in corr_dict:
    corr_dict[key] = [x for x in corr_dict[key] if x == x]

with open("./planner/Frenet/results/corr_dict.json", "w") as fp:
    json.dump(corr_dict, fp)

with open("./planner/Frenet/results/scen_list.txt", "w") as fp2:
    json.dump(scenario_list, fp2)

clean_corr_dict = {k: corr_dict[k] for k in corr_dict if not isnan(sum(corr_dict[k]))}

fig, ax = plt.subplots()
ax.boxplot(clean_corr_dict.values())
ax.set_xticklabels(clean_corr_dict.keys(), rotation=90)

plt.tight_layout()
plt.savefig("./planner/Frenet/results/correlations.pdf")


print("Done.")
