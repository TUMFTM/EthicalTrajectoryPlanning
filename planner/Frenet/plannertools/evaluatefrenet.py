"""Evaluate a frenet Planner."""
import sys
import pathlib

if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[4]))

    from EthicalTrajectoryPlanning.planner.plannertools.evaluate import (
        ScenarioEvaluator,
        DatasetEvaluator,
    )
    from EthicalTrajectoryPlanning.planner.Frenet.plannertools.frenetcreator import FrenetCreator
    from EthicalTrajectoryPlanning.planner.Frenet.configs.load_json import load_planning_json, load_weight_json, load_risk_json


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="ethical")
    parser.add_argument('--settings', default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    # load settings from planning.json and weights.json
    settings_dict = load_planning_json()
    eval_directory = (
        pathlib.Path(__file__).resolve().parents[1].joinpath("results").joinpath("eval")
    )
    weights = load_weight_json(filename=f"weights_{args.weights}.json")
    if args.settings is None:
        risk_dict = load_risk_json()
    else:
        risk_dict = load_risk_json(filename=f"risk_{args.settings}.json")
    settings_dict["risk_dict"] = risk_dict

    # Create the frenet creator
    frenet_creator = FrenetCreator(settings_dict, weights=weights)

    # Create the scenario evaluator
    evaluator = ScenarioEvaluator(
        planner_creator=frenet_creator,
        vehicle_type=settings_dict["evaluation_settings"]["vehicle_type"],
        path_to_scenarios=pathlib.Path("../commonroad-scenarios/scenarios/").resolve(),
        log_path=pathlib.Path("./log/example").resolve(),
        collision_report_path=eval_directory,
        timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"],
    )
    # Create the Dataset Evaluator
    limit_scenarios = (None if args.all else 1)
    dataset_evaluator = DatasetEvaluator(
        evaluator, eval_directory, limit_scenarios=limit_scenarios, disable_mp=False
    )
    # Eval the dataset
    dataset_evaluator.eval_dataset()
