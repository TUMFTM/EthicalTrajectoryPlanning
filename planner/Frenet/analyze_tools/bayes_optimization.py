#!/user/bin/env python

"""Script to optimize the cost function weights of the frenét planner."""

import os
import sys
from joblib import Parallel, delayed
import multiprocessing
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import numpy as np

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(module_path)

from planner.Frenet.evaluate import evaluate_scenario


def optimize_weights(
    ego_risk: float = 0.0, velocity: float = 0.0, dist_to_global_path: float = 0.0
):
    """
    Optimize the weights of the cost functions given as arguments.

    Args:
        ego_risk (float): Weight of the cost function that minimizes the ego risk. Defaults to 0.0.
        velocity (float): Weight of the cost function that chooses a good velocity. Defaults to 0.0.
        dist_to_global_path (float): Weight of the cost function that minimizes the distance to the global path. Defaults to 0.0.

    Returns:
        int: Number of successfully solved scenarios.
    """
    # create the weights of the cost function
    cost_weights = {
        "ego_risk": ego_risk,
        "max_risk": 0.0,
        "visible_area": 0.0,
        "lon_jerk": 0.0,
        "lat_jerk": 0.0,
        "velocity": velocity,
        "dist_to_global_path": dist_to_global_path,
        "travelled_dist": 0.0,
        "dist_to_goal_pos": 0.0,
        "dist_to_lane_center": 0.0,
    }

    # e. g. run the optimization with 3 scenarios
    # just to test it, for a real optimization more scenarios would be necessary
    scenario_names = [
        "hand-crafted/DEU_A9-1_1_T-1.xml",
        "hand-crafted/DEU_A9-2_1_T-1.xml",
        "THI-Bicycle/RUS_Bicycle-4_1_T-1.xml",
    ]

    # create the paths of the scenarios
    for i in range(len(scenario_names)):
        scenario_names[i] = os.path.abspath(
            os.path.join(
                os.path.abspath(__file__),
                "../../../../commonroad-scenarios/scenarios/",
                scenario_names[i],
            )
        )

    # set the arguments of the used motion planner
    raise DeprecationWarning("Use the parameters from the json file.")
    arg_dict = {
        "vehicle_type": "bmw_320i",
        "timing": False,
        "save_animation": False,
        "show_visualization": False,
        "save_solution": False,
        "mode": 2,
        "cost_function_weight": cost_weights,
        "frenet_parameters": {
            "t_list": [2.0],
            "v_list_generation_mode": 0,
            "n_v_samples": 5,
            "d_list": np.linspace(-3.5, 3.5, 15),
            "dt": 0.1,
            "v_thr": 3.0,
        },
    }

    # create arg_list (scenario_name and arg_dict)
    arg_list = [[scenario_names[i], arg_dict] for i in range(len(scenario_names))]

    # Evaluate scenes on all available cores
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(evaluate_scenario)(i) for i in arg_list
    )

    # get number of successfully solved scenarios
    n_success = 0
    for i in range(len(results)):
        if results[i]["reason_for_failure"] == "Goal reached!":
            n_success += 1

    # return the number of successfully solved scenarios (this value should be optimized)
    return n_success


if __name__ == "__main__":
    # Path for logs
    log_path = "./planner/Frenet/bayes_logs.json"

    # Bounded region of parameter space
    pbounds = {
        "ego_risk": (0, 10),
        # 'max_risk': (0, 10),
        # 'lon_jerk': (0, 10),
        # 'lat_jerk': (0, 10),
        "velocity": (0, 10),
        "dist_to_global_path": (0, 10),
        # 'travelled_dist': (0, 10),
        # 'dist_to_goal_pos': (0, 10),
        # 'dist_to_lane_center': (0, 10),
    }

    # initialize the bayesian optimization
    optimizer = BayesianOptimization(
        f=optimize_weights, pbounds=pbounds, random_state=1, verbose=2
    )

    # give the optimizes an initial guess
    optimizer.probe(
        params={"ego_risk": 0.0, "velocity": 1.0, "dist_to_global_path": 4.0}, lazy=True
    )

    # load already done iterations if available
    if os.path.exists(log_path):
        load_logs(optimizer, logs=[log_path])

    # log the results
    logger = JSONLogger(path=log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # start the optimization
    optimizer.maximize(init_points=5, n_iter=10)

# EOF
