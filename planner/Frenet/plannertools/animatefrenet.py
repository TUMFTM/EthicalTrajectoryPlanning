"""Module for animating the frenet planner."""

import pathlib
import sys

if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[4]))


from EthicalTrajectoryPlanning.planner.Frenet.configs.load_json import load_planning_json
from EthicalTrajectoryPlanning.planner.plannertools.animate import ScenarioAnimator
from EthicalTrajectoryPlanning.planner.Frenet.plannertools.frenetcreator import FrenetCreator


if __name__ == "__main__":
    # EXAMPLE USAGE
    # CURRENTLY NOT IMPLEMENTED COMPLETELY

    # load settings from planning.json
    settings_dict = load_planning_json()

    frenet_creator = FrenetCreator(settings_dict)

    eval_directory = (
        pathlib.Path(__file__).resolve().parents[1].joinpath("results").joinpath("eval")
    )

    animator = ScenarioAnimator(
        planner_creator=frenet_creator,
        vehicle_type=settings_dict["evaluation_settings"]["vehicle_type"],
        path_to_scenarios=pathlib.Path("../commonroad-scenarios/scenarios/").resolve(),
        log_path=pathlib.Path("./log/example").resolve(),
        timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"],
        collision_report_path=eval_directory,
    )
    animator.animate_scenario("hand-crafted/DEU_A99-1_1_T-1.xml")
