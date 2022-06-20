"""Analyze directory of logfiles for risk distributions."""
import os
import argparse
import json
import progressbar
import traceback
from analyze_log import FrenetLogVisualizer

VUL_ROAD_USERS = ["PEDESTRIAN", "BICYCLE", "MOTORCYCLE"]


def listdir_fullpath(d):
    """Get listdir with full paths.

    Args:
        d ([str]): [directory]

    Returns:
        [list]: [list with full paths of containing files]
    """
    return [os.path.join(d, f) for f in os.listdir(d)]


def get_risk_list(logdir):
    """Extract risk lists for ego, 3rd party and vulnerables out of logfiles in a given logdir.

    Args:
        logdir ([str]): [logdir with logfiles in it]

    Returns:
        [tuple of lists]: [list of ego, 3rd party and vul risks]
    """
    log_file_list = listdir_fullpath(logdir)
    risk_list_ego = []
    risk_list_3rd_party = []
    risk_list_vul = []
    with progressbar.ProgressBar(max_value=len(log_file_list)).start() as pbar:
        for logfile in log_file_list:
            try:
                logvisualizer = FrenetLogVisualizer(logfile, visualize=False)

                for traj in logvisualizer.best_traj_list:
                    risk_list_ego.extend(list(traj["ego_risk_dict"].values()))
                    for obst in traj["obst_risk_dict"]:
                        # check for vulnerable road users
                        if (
                            logvisualizer.scenario.obstacle_by_id(
                                int(obst)
                            ).obstacle_type.name
                            in VUL_ROAD_USERS
                        ):
                            risk_list_vul.append(traj["obst_risk_dict"][obst])
                    risk_list_3rd_party.extend(list(traj["obst_risk_dict"].values()))
            except Exception:
                print(traceback.format_exc())

            pbar.update(pbar.value + 1)

    return risk_list_ego, risk_list_3rd_party, risk_list_vul


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./planner/Frenet/results/logs")
    args = parser.parse_args()

    risk_list_ego, risk_list_3rd_party, risk_list_vul = get_risk_list(args.logdir)

    risk_values = {
        os.path.dirname(args.logdir).split("/")[-1] + "_ego": risk_list_ego,
        os.path.dirname(args.logdir).split("/")[-1] + "_3rd": risk_list_3rd_party,
        os.path.dirname(args.logdir).split("/")[-1] + "_vul": risk_list_vul,
    }

    with open(
        os.path.join(os.path.dirname(args.logdir), "risk_values.json"), "w"
    ) as fp:
        json.dump(risk_values, fp)

# EOF
