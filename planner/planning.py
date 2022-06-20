from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

from agent_sim.agent import Agent
import numpy as np
import os
import sys

module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(module_path)

from commonroad_helper_functions.exceptions import (
    GoalReachedNotification,
    NoGlobalPathFoundError,
    ScenarioCompatibilityError,
)
from planner.utils.goalcheck import GoalReachedChecker
from planner.GlobalPath.lanelet_based_planner import LaneletPathPlanner
from planner.utils.timers import ExecTimer
from planner.Frenet.utils.helper_functions import get_max_curvature
from planner.Frenet.utils.calc_trajectory_cost import distance
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D


class PlanningAgent(Agent):
    def __init__(
        self,
        scenario: Scenario,
        agent_id: int,
        predictor=None,
        planner=None,
        control_dynamics=None,
        enable_logging: bool = True,
        log_path: str = "/log",
        debug_step: bool = False,
    ):

        # initialize parent class (Agent)
        super().__init__(
            scenario=scenario,
            agent_id=agent_id,
            enable_logging=enable_logging,
            log_path=log_path,
            debug_step=debug_step,
        )

        self.__control_dynamics = control_dynamics

        # predictor
        self.__predictor = predictor

        # planner
        self.__planner = planner

        # additional state variables:
        # s: arc-length:
        # d: lateral offset from reference spline
        # d_d: lateral velocity
        # d_dd: lateral acceleration
        self._s = 0.0
        self._d = 0.0
        self._d_d = 0.0
        self._d_dd = 0.0

        # missing initial state variables
        if not hasattr(self.state, "acceleration"):
            self._state.acceleration = 0.0

    def _step_agent(self, delta_time):
        """Step Agent

        This is the step function for a planning agent.
        It first calls the predictor, than the planner and calculates the new state according to the specified controller and dynamic model.

        :param delta_time: time since previous step
        """
        if self.predictor is not None:
            # with prediction
            self.predictor.step(
                scenario=self.scenario,
                time_step=self.time_step,
                obstacle_id_list=list(self.scenario.dynamic_obstacles.keys()),
                multiprocessing=False,
            )

            prediction = self.predictor.prediction_result
        else:
            # without prediction
            prediction = None

        self.planner.step(
            scenario=self.scenario,
            current_lanelet_id=self.current_lanelet_id,
            time_step=self.time_step,
            ego_state=self.state,
            prediction=prediction,
        )

        if self.control_dynamics is not None:
            self._state = self.control_dynamics.step(self.planner.trajectory)

        else:
            # assume ideal tracking
            # the next state is the 2nd entry of the trajectory
            if self.planner.trajectory["time_s"][1] != self.dt:
                # current assumption: a trajectory is planned every time step
                raise ScenarioCompatibilityError(
                    "The scenario time step size does not match the time discretization of "
                    "the trajectory or the replanning frequency."
                )
            else:
                # assuming that the trajectory is discretized with the scenario time step size and replanned every time step
                i = 1
                self._state.position = np.array(
                    [
                        self.planner.trajectory["x_m"][i],
                        self.planner.trajectory["y_m"][i],
                    ]
                )
                self._state.orientation = self.planner.trajectory["psi_rad"][i]
                self._state.velocity = self.planner.trajectory["v_mps"][i]
                self._state.acceleration = self.planner.trajectory["ax_mps2"][i]

                self._s = self.planner.trajectory["s_loc_m"][i]
                self._d = self.planner.trajectory["d_loc_m"][i]
                self._d_d = self.planner.trajectory["d_d_loc_mps"][i]
                self._d_dd = self.planner.trajectory["d_dd_loc_mps2"][i]

    @property
    def predictor(self):
        """Predictor of the planning agent"""
        return self.__predictor

    @property
    def planner(self):
        """Planner of the planning agent"""
        return self.__planner

    @property
    def control_dynamics(self):
        """Controller and dynamic model of the planning agent"""
        return self.__control_dynamics


def add_ego_vehicles_to_scenario(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, vehicle_params
):
    """Add Ego Vehicle to Scenario

    This function adds a ego vehicle represented by a dynamic obstacle for each planning problem specified in the planning problem set

    :param scenario: commonroad scenario
    :param planning_problem_set: commonroad planning problem set
    :return: new scenario, dictionary with agent IDs as key and planning problem ID as value
    """

    # dictionary to gather all ego IDs and assign them to a planning problem
    agent_planning_problem_id_assignment = {}
    for (
        planning_problem_id,
        planning_problem,
    ) in planning_problem_set.planning_problem_dict.items():
        # Obstacle shape
        # TODO: Implement geometric parameters like length and width

        obstacle_shape = Rectangle(length=vehicle_params.l, width=vehicle_params.w)

        # Create a dynamic obstacle in the scenario
        agent_id = scenario.generate_object_id()

        agent_planning_problem_id_assignment[agent_id] = planning_problem_id

        trajectory = Trajectory(
            initial_time_step=1, state_list=[planning_problem.initial_state]
        )

        prediction = TrajectoryPrediction(trajectory=trajectory, shape=obstacle_shape)

        ego_obstacle = DynamicObstacle(
            obstacle_id=agent_id,
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=obstacle_shape,
            initial_state=planning_problem.initial_state,
            prediction=prediction,
        )

        # add object to scenario
        scenario.add_objects(ego_obstacle)

    return scenario, agent_planning_problem_id_assignment


class Planner(object):
    """Main Planner Class"""

    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        ego_id: int,
        vehicle_params,
        exec_timer=None,
    ):

        # commonroad scenario
        self.__scenario = scenario

        # current lanelet the planning vehicle is moving on
        self.__current_lanelet_id = None

        # planning problem for the planner
        self.__planning_problem = planning_problem
        self.__goal_checker = GoalReachedChecker(planning_problem)

        # initial time step
        self.__time_step = 0

        # ID of the planning vehicle
        self.__ego_id = ego_id

        # initial state of the planning vehicle
        self.__ego_state = planning_problem.initial_state

        # minimum trajectory length
        self.__min_trajectory_length = 50

        # prediction
        self.__prediction = None

        # Timer for timing execution times.
        self.__exec_timer = (
            ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer
        )

        # Variables that contain the global path
        # TODO Remove everything except the reference spline.....
        # NOTE They are all referenced within the Frenet Planner -> tricky to remove....
        self.global_path_length = None
        self.global_path = None
        self.global_path_to_goal = None
        self.global_path_after_goal = None
        self.__reference_spline = None
        self.plan_global_path(scenario, planning_problem, vehicle_params)

        # trajectory
        dt = 0.1
        self._trajectory = {
            "s_loc_m": np.zeros(self.min_trajectory_length),
            "d_loc_m": np.zeros(self.min_trajectory_length),
            "d_d_loc_mps": np.zeros(self.min_trajectory_length),
            "d_dd_loc_mps2": np.zeros(self.min_trajectory_length),
            "x_m": np.zeros(self.min_trajectory_length),
            "y_m": np.zeros(self.min_trajectory_length),
            "psi_rad": np.zeros(self.min_trajectory_length),
            "kappa_radpm": np.zeros(self.min_trajectory_length),
            "v_mps": np.zeros(self.min_trajectory_length),
            "ax_mps2": np.zeros(self.min_trajectory_length),
            "time_s": np.arange(0, dt * self.min_trajectory_length, dt),
        }

    def step(
        self,
        scenario: Scenario,
        current_lanelet_id: int,
        time_step: int,
        ego_state: State,
        prediction=None,
        v_max=50,
    ):
        """Main Step Function of the Planner

        This method generates a new trajectory for the current scenario und prediction.
        It is a wrapper for the planner-type depending actual step method "_step_planner()" that updates the trajectory.
        "_step_planner()" must be overloaded by inheriting classes that implement planning methods.

        :param scenario: commonroad scenario
        :param current_lanelet_id: current lanelet id of the planning vehicle
        :param time_step: current time step
        :param ego_state: current state of the planning vehicle
        :param prediction: prediction
        :param v_max: maximum allowed velocity on the current lanelet in m/s
        """
        self.__scenario = scenario
        self.__current_lanelet_id = current_lanelet_id
        self.__time_step = time_step
        self.__ego_state = ego_state
        self.__prediction = prediction

        # TODO: Include maximum allowed speed
        self.__v_max = v_max

        # Check if the goal is alreay reached
        self.__check_goal_reached()

        # call the planner-type depending step function to generate a new trajectory
        self._step_planner()

    def _step_planner(self):
        """Planner step function

        This method directyl changes the planne trajectory. It must be overloaded by an inheriting planner class.
        There is no basic trajectory planning implemented.
        """
        raise NotImplementedError(
            "No basic trajectory planning implemented. "
            "Overload the method _step_planner() to generate a trajectory."
        )

    def __check_goal_reached(self):
        # Get the ego vehicle
        self.goal_checker.register_current_state(self.ego_state)
        if self.goal_checker.goal_reached_status():
            raise GoalReachedNotification("Goal reached in time!")
        elif self.goal_checker.goal_reached_status(ignore_exceeded_time=True):
            raise GoalReachedNotification("Goal reached but time exceeded!")

    def plan_global_path(self, scenario, planning_problem, vehicle_params, initial_state=None):
        """Plan a global path to the planning's problem target area.

        Args:
            scenario (_type_): _description_
            planning_problem (_type_): _description_
            vehicle_params (_type_): _description_
            initial_state (_type_, optional): _description_. Defaults to None.

        Raises:
            NoGlobalPathFoundError: _description_
        """
        with self.exec_timer.time_with_cm("initialization/plan global path"):
            # calculate the global path for the planning problem
            try:
                # if the velocity is pretty high, increase the max length for a lane change

                global_path_max_lane_change_length = (
                    self.ego_state.velocity * 3.0
                    if self.ego_state.velocity > 15
                    else 20
                )
                path_planner = LaneletPathPlanner(
                    scenario=scenario,
                    planning_problem=planning_problem,
                    max_lane_change_length=global_path_max_lane_change_length,
                    initial_state=initial_state
                )
                (
                    initial_global_path,
                    self.global_path_length,
                ) = path_planner.plan_global_path()
            # raise error if global path planner fails
            except TypeError:
                raise NoGlobalPathFoundError(
                    "Failed. Could not find a global path for the planning problem."
                ) from None

        with self.exec_timer.time_with_cm("initialization/check curvature of path"):
            # check the curvature of the global path
            self.global_path = check_curvature_of_global_path(
                global_path=initial_global_path,
                planning_problem=planning_problem,
                vehicle_params=vehicle_params,
                ego_state=self.ego_state,
            )

            # split the global path for visualization purposes
            (
                self.global_path_to_goal,
                self.global_path_after_goal,
            ) = path_planner.split_global_path(global_path=self.global_path)

            # create the reference spline from the global path
            self.__reference_spline = CubicSpline2D(
                x=self.global_path[:, 0], y=self.global_path[:, 1]
            )

    @property
    def planning_problem(self):
        """Planning problem to be solved"""
        return self.__planning_problem

    @property
    def goal_checker(self):
        """Return the goal checker."""
        return self.__goal_checker

    @property
    def exec_timer(self):
        """Return the exec_timer object."""
        return self.__exec_timer

    @property
    def reference_spline(self):
        """Return the reference spline object."""
        return self.__reference_spline

    @property
    def scenario(self):
        """Commonroad scenario"""
        return self.__scenario

    @property
    def time_step(self):
        """Current time step"""
        return self.__time_step

    @property
    def ego_id(self):
        """ID of the planning vehicle"""
        return self.__ego_id

    @property
    def ego_state(self):
        """Current state of the planning vehicle"""
        return self.__ego_state

    @property
    def min_trajectory_length(self):
        """Minimum length of the planned trajectory"""
        return self.__min_trajectory_length

    @property
    def trajectory(self):
        """Planned trajectory"""
        return self._trajectory

    @property
    def prediction(self):
        """Prediction"""
        return self.__prediction

    @property
    def v_max(self):
        """maximum velocity"""
        return self.__v_max

    @property
    def current_lanelet_id(self):
        """Current lanelet"""
        return self.__current_lanelet_id


# TODO move to separate file
def check_curvature_of_global_path(
    global_path: np.ndarray, planning_problem, vehicle_params, ego_state
):
    """
    Check the curvature of the global path.

    If the curvature is to high, points of the global path are removed to smooth the global path. In addition, a new point is added which ensures the initial orientation.

    Args:
        global_path (np.ndarray): Coordinates of the global path.

    Returns:
        np.ndarray: Coordinates of the new, smooth global path.

    """
    global_path_curvature_ok = False

    # get start velocity of the planning problem
    start_velocity = planning_problem.initial_state.velocity

    # calc max curvature for the initial velocity
    max_initial_curvature, _ = get_max_curvature(
        vehicle_params=vehicle_params, v=start_velocity
    )

    # get x and y from the global path
    global_path_x = global_path[:, 0].tolist()
    global_path_y = global_path[:, 1].tolist()

    # add a point to the global path to ensure the initial orientation of the planning problem
    # never delete this point or the initial point
    new_x = ego_state.position[0] + np.cos(ego_state.orientation) * 0.1
    new_y = ego_state.position[1] + np.sin(ego_state.orientation) * 0.1
    global_path_x.insert(1, new_x)
    global_path_y.insert(1, new_y)

    # check if the curvature of the global path is ok
    while global_path_curvature_ok is False:
        # calc the already covered arc length for the points of global path
        global_path_s = [0.0]

        for i in range(len(global_path_x) - 1):
            p_start = np.array([global_path_x[i], global_path_y[i]])
            p_end = np.array([global_path_x[i + 1], global_path_y[i + 1]])
            global_path_s.append(distance(p_start, p_end) + global_path_s[-1])

        # calculate the curvature of the global path
        dx = np.gradient(global_path_x, global_path_s)
        dy = np.gradient(global_path_y, global_path_s)

        ddx = np.gradient(dx, global_path_s)
        ddy = np.gradient(dy, global_path_s)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

        # loop through every curvature of the global path
        global_path_curvature_ok = True
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            if (curvature[i] * 2.0) > max_initial_curvature:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner itself
                if global_path_s[index_closest_path_point] <= 10.0:
                    global_path_x.pop(index_closest_path_point)
                    global_path_y.pop(index_closest_path_point)
                    global_path_curvature_ok = False
                    break

        # also check if the curvature is smaller than the turning radius anywhere
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            if (curvature[i] * 2.0) > get_max_curvature(
                vehicle_params=vehicle_params, v=0.0
            )[0]:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner itself
                global_path_x.pop(index_closest_path_point)
                global_path_y.pop(index_closest_path_point)
                global_path_curvature_ok = False
                break

    # create the new global path
    new_global_path = np.array([np.array([global_path_x[0], global_path_y[0]])])
    for i in range(1, len(global_path_y)):
        new_global_path = np.concatenate(
            (
                new_global_path,
                np.array([np.array([global_path_x[i], global_path_y[i]])]),
            )
        )

    return new_global_path


# EOF
