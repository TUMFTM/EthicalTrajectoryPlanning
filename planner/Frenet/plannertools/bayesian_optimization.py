"""Optimize the weights of the cost function with BO."""
import json
import sys
import pathlib
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from planner.plannertools.evaluate import (
    DatasetEvaluator,
    EvalFilesGenerator,
    ScenarioEvaluator,
)
from planner.Frenet.plannertools.frenetcreator import FrenetCreator
from planner.Frenet.configs.load_json import load_planning_json
from sklearn import preprocessing

if __name__ == "__main__":
    # to be removed
    eval_directory = (
        pathlib.Path(__file__).resolve().parents[1].joinpath("results").joinpath("eval")
    )

    # get the results

    def load_results_vel():
        """Load the results from the planner_statistic."""
        planner_statistic = eval_directory / 'planner_statistic.log'
        with open(planner_statistic, 'r') as f:
            lines = f.readlines()
            avg_vel = lines[7]
            avg_vel = avg_vel.split()
            avg_vel = avg_vel[2]
            return float(avg_vel)

    def load_results_harm():
        """Load the harm from harm.json."""
        harm = eval_directory / 'harm.json'
        with open(harm, 'r') as f:
            harm = json.load(f)
            harm = harm['Total']
        return harm

    def update_weights():
        """Update the new weights to a dict."""
        weights_bayes = {}
        weights_bayes["bayes"] = float(25)
        weights_bayes["equality"] = float(25)
        weights_bayes["maximin"] = float(0.0)
        weights_bayes["ego"] = float(0.0)
        weights_bayes["risk_cost"] = float(0.0)
        weights_bayes["visible_area"] = float(candidate[0])
        weights_bayes["lon_jerk"] = float(candidate[1])
        weights_bayes["lat_jerk"] = float(candidate[2])
        weights_bayes["velocity"] = float(candidate[3])
        weights_bayes["dist_to_global_path"] = float(candidate[4])
        weights_bayes["travelled_dist"] = float(candidate[5])
        weights_bayes["dist_to_goal_pos"] = float(candidate[6])
        weights_bayes["dist_to_lane_center"] = float(candidate[7])
        return weights_bayes

    # define the input
    # x_dict = frenet_planner.load_weight_json()
    # not_used_weights = ['bayes', 'equality', 'maximin', 'ego', 'risk_cost']
    # x = []
    # for key in not_used_weights:
    #   del x_dict[key]
    # for key in x_dict:
    #    x.append(x_dict[key])

    # just for testing
    y = [[2], [2.1]]

    x = [
        [0.0, 0.0, 0.5, 0.6, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.7, 0.8, 0.0, 0.0, 1, 0.0],
    ]

    x_best = [0.0, 0.0, 0.5, 0.6, 0.0, 0.0, 0.5, 0.0]
    number = 0  # only used for the overview output

    # parameters for the BO
    b = [[-6, -6, -6, -6, -6, -6, -6, -6], [6, 6, 6, 6, 6, 6, 6, 6]]
    bounds = torch.FloatTensor(b)

    for i in range(10):
        # standardize the data
        x_standardized = preprocessing.scale(x, axis=1)
        y_standardized = preprocessing.scale(y, axis=1)
        train_x = torch.FloatTensor(x_standardized)
        train_y = torch.FloatTensor(y_standardized)

        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        best_f = torch.max(train_y)
        EI = ExpectedImprovement(gp, best_f)

        candidate, acq_value = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        candidate = candidate.numpy()
        candidate = candidate[0]
        candidate_list = candidate.tolist()
        x.append(candidate_list)

        # calculate the result
        sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

        # evaluate the scenario(s) with the current setting
        # load settings from planning.json
        settings_dict = load_planning_json()
        eval_directory = (
            pathlib.Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("results")
            .joinpath("eval")
        )
        # Create the frenet creator
        frenet_creator = FrenetCreator(settings_dict)

        # Create the scenario evaluator
        evaluator = ScenarioEvaluator(
            planner_creator=frenet_creator,
            vehicle_type=settings_dict["evaluation_settings"]["vehicle_type"],
            path_to_scenarios=pathlib.Path(
                "../commonroad-scenarios/scenarios/"
            ).resolve(),
            log_path=pathlib.Path("./log/example").resolve(),
            collision_report_path=eval_directory,
            timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"],
        )
        # Create the Dataset Evaluator
        dataset_evaluator = DatasetEvaluator(
            evaluator,
            eval_directory,
            ['hand-crafted/ZAM_Tjunction-1_6_T-1.xml'],
            disable_mp=False,
        )

        # Update the weights
        frenet_creator.weights = update_weights()

        # Eval the dataset
        dataset_evaluator.eval_dataset()

        # Load the velocity
        # Note: change for weighted or driven velocity
        y_iteration = EvalFilesGenerator(dataset_evaluator)._avg_velocity
        y.append([y_iteration])

        if y_iteration > best_f:
            x_best = candidate_list
            number = i
        print(
            f"the {i}th candidate is {candidate}, the best value is {best_f} at {x_best} found in iteration {number}"
        )
