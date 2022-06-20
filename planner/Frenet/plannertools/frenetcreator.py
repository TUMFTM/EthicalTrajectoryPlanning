"""Evaluate a frenet Planner."""

from EthicalTrajectoryPlanning.planner.plannertools.scenario_handler import PlannerCreator
from EthicalTrajectoryPlanning.planner.Frenet.frenet_planner import FrenetPlanner


class FrenetCreator(PlannerCreator):
    """Class for constructing a planner object from a Handler object."""

    def __init__(self, settings, weights=None):
        """__init__ function to construct the object.

        This function is called from the user.
        """
        # Settings specific for a frenet planner.
        self.show_visualization = settings["evaluation_settings"]["show_visualization"]
        self.frenet_settings = settings["frenet_settings"]
        self.weights = weights

    def get_planner(self, scenario_handler, ego_vehicle_id):
        """Create the planner from the scenario handler object.

        Args:
            scenariWo_handler (obj): scenario handler object

        Raises:
            NotImplementedError: Abstract Method

        Returns:
            obj: a planner object.
        """
        return FrenetPlanner(
            scenario=scenario_handler.scenario,
            planning_problem=scenario_handler.planning_problem_set.find_planning_problem_by_id(
                scenario_handler.agent_planning_problem_id_assignment[ego_vehicle_id]
            ),
            ego_id=ego_vehicle_id,
            vehicle_params=scenario_handler.vehicle_params,
            exec_timer=scenario_handler.exec_timer,
            mode=self.frenet_settings["mode"],
            plot_frenet_trajectories=self.show_visualization,
            frenet_parameters=self.frenet_settings["frenet_parameters"],
            weights=self.weights,
        )

    @staticmethod
    def get_blacklist():
        """Return the scenario blacklist for this planner."""
        bad_scenario_names = [
            "ARG",
            "Luckenwalde",
            "USA_US101-33_1_T-1",
            "USA_US101-22_2_T-1",
            "USA_US101-3_2_S-1",
            "USA_US101-9_4_T-1",
            "USA_US101-1_1_T-1",
            "USA_US101-1_1_S-1",
            "DEU_Hhr-1_1",
            "interactive",
        ]
        set_based_scenarios = [
            "DEU_Ffb-1_2_S-1",
            "DEU_Ffb-2_2_S-1",
            "DEU_Muc-30_1_S-1",
            "USA_Lanker-1_1_S-1",
            "USA_US101-1_1_S-1",
            "USA_US101-1_2_S-1",
            "USA_US101-2_2_S-1",
            "USA_US101-2_3_S-1",
            "USA_US101-2_4_S-1",
            "USA_US101-3_2_S-1",
            "USA_US101-3_3_S-1",
            "USA_US101-7_1_S-1",
            "USA_US101-7_2_S-1",
            "ZAM_ACC-1_2_S-1",
            "ZAM_ACC-1_3_S-1",
            "ZAM_HW-1_1_S-1",
            "ZAM_Intersect-1_1_S-1",
            "ZAM_Intersect-1_2_S-1",
            "ZAM_Urban-1_1_S-1",
            "ZAM_Urban-4_1_S-1",
            "ZAM_Urban-5_1_S-1",
            "ZAM_Urban-6_1_S-1",
            "ZAM_Urban-7_1_S-1",
        ]
        return bad_scenario_names + set_based_scenarios
