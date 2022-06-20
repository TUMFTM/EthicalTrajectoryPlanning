"""Module for creating a animation of a scenario driven by a planner."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from EthicalTrajectoryPlanning.planner.plannertools.scenario_handler import ScenarioHandler
from commonroad_helper_functions.customvehicleicons import draw_obstacle
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad_helper_functions.exceptions import GoalReachedNotification


class ScenarioAnimator(ScenarioHandler):
    """Class for animating a scenario."""

    def __init__(self, *args, **kwargs):
        """WORKING STATE. CHANGE IN PROGRESS."""
        super().__init__(*args, **kwargs)
        self.fig_size = 7  # in inches
        self.fig = plt.figure(figsize=(self.fig_size, self.fig_size))
        self.export_res = 1080  # 1080p
        self.view_size = 64
        self.draw_planning_problem = False
        self.mode = "show"
        self.export_path = pathlib.Path("./animations").resolve()

    def draw_current_frame(self, **kwargs):
        """WORKING STATE. CHANGE IN PROGRESS."""
        plt.gcf().gca().cla()
        agent = kwargs["agent"]
        planner = agent.planner
        ego_vehicle_id = planner.ego_id
        ego_vehicle = self.scenario.obstacle_by_id(ego_vehicle_id)
        current_pos = ego_vehicle.state_at_time(kwargs["time_step"]).position
        plot_limits = [
            current_pos[0] - self.view_size / 2,
            current_pos[0] + self.view_size / 2,
            current_pos[1] - self.view_size / 2,
            current_pos[1] + self.view_size / 2,
        ]
        draw_object(
            self.scenario,
            draw_params={
                "time_begin": kwargs["time_step"],
                "dynamic_obstacle": {"draw_shape": False},
            },
            plot_limits=plot_limits,
        )
        if self.draw_planning_problem:
            draw_object(self.planning_problem_set, plot_limits=plot_limits)

        self._draw_planned_trajectory(planner)

        self._draw_driven_trajectory(ego_vehicle)

        self._draw_obstacles(kwargs["time_step"])

        plt.gca().axis("equal")
        plt.gca().set_xlim(plot_limits[0], plot_limits[1])
        plt.gca().set_ylim(plot_limits[2], plot_limits[3])

    def _draw_planned_trajectory(self, planner):
        plt.plot(
            planner.trajectory["x_m"],
            planner.trajectory["y_m"],
            "g-",
            zorder=30,
            markersize=2,
        )

    def _draw_driven_trajectory(self, ego_vehicle):
        driven_trajectory = np.array(
            [state.position for state in ego_vehicle.prediction.trajectory.state_list]
        )
        plt.plot(
            driven_trajectory[:, 0],
            driven_trajectory[:, 1],
            "-",
            color="orange",
            zorder=30,
            markersize=2,
        )

    def _draw_obstacles(self, time_step):
        for obstacle in self.scenario.dynamic_obstacles:
            draw_obstacle(obstacle, time_step, self.fig.gca())

    def _do_simulation_step(self, **kwargs):
        super()._do_simulation_step(**kwargs)
        self.draw_current_frame(**kwargs)
        if self.mode == "show":
            plt.pause(0.001)
        if self.mode == "export":
            agent_id = kwargs["agent"].agent_id
            frame_path = (
                self.export_path.joinpath("frames")
                .joinpath("visualization")
                .joinpath(f"agent-id-{str(agent_id)}")
                .joinpath(str(kwargs["time_step"]))
                .with_suffix(".png")
            )
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.subplots_adjust(left=0.06, bottom=0.04, right=0.995, top=0.995)
            self.fig.savefig(
                frame_path,
                dpi=int(self.export_res / self.fig_size),
            )

    def animate_scenario(self, scenario_path):
        """Has to be implemented in the future."""
        self.scenario_path = self.path_to_scenarios.joinpath(scenario_path)
        try:
            self._initialize()
            self._simulate()
        except GoalReachedNotification:
            print("Scenario completed successfully")
