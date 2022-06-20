"""Module for creating a solution file of a scenario driven by a planner."""


from planner.plannertools.scenario_handler import ScenarioHandler


class ScenarioSolutionWriter(ScenarioHandler):
    """Class for writing a solution file."""

    def write_solution_file(self, scenario_path):
        """Has to be implemented in the future."""
