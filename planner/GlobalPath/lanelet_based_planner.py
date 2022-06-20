#!/user/bin/env python

"""Implementation of a lanelet based global path planner."""

# Standard imports
import os
import sys
import time
import math
from typing import List

# Third Party imports
import matplotlib.pyplot as plt
import numpy as np

# Custom imports
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import Interval
from commonroad.scenario.lanelet import Lanelet
from commonroad.visualization.draw_dispatch_cr import draw_object


module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(module_path)

from planner.GlobalPath.utils.quintic_polynomials_planner import (
    quintic_polynomials_planner,
)


__author__ = "Florian Pfab"
__email__ = "Florian.Pfab@tum.de"
__date__ = "23.05.2020"


# Class definition
class LaneletPathPlanner:
    """Class of the lanelet based path planner."""

    def __init__(
        self,
        scenario,
        planning_problem,
        print_info: bool = False,
        max_lane_change_length: float = 10.0,
        initial_state=None,
    ):
        """
        Initialize a lanelet based path planner.

        Args:
            scenario (Scenario): Environment of the path planner described by the scenario.
            planning_problem (PlanningProblem): Planning problem which should be solved.
            print_info (bool): Show basic information about the planning. Defaults to False.
            max_lane_change_length (float, optional): Maximum distance a lane change should take. Defaults to 10.0.

        """
        # Store input parameters
        self.scenario = scenario
        self.planningProblem = planning_problem

        # Create necessary attributes
        self.lanelet_network = self.scenario.lanelet_network
        if initial_state:
            self.initial_state = initial_state
        else:
            self.initial_state = self.planningProblem.initial_state

        # Get lanelet id of the starting lanelet (of initial state)
        self.startLanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position(
            [self.initial_state.position]
        )[0]

        self.goalLanelet_ids = []
        # Get the goal lanelet ids, if they are given directly in the planning problem
        if (
            hasattr(self.planningProblem.goal, 'lanelets_of_goal_position')
            and self.planningProblem.goal.lanelets_of_goal_position is not None
        ):
            self.goalLanelet_ids = self.planningProblem.goal.lanelets_of_goal_position[
                0
            ]
        else:
            # Get lanelet id of the ending lanelet (of goal state),this depends on type of goal state
            if hasattr(self.planningProblem.goal.state_list[0], 'position'):
                if hasattr(self.planningProblem.goal.state_list[0].position, 'center'):
                    self.goalLanelet_ids = (
                        self.scenario.lanelet_network.find_lanelet_by_position(
                            [self.planningProblem.goal.state_list[0].position.center]
                        )[0]
                    )

        # If the planning Problem has no goal position, interrupt here
        if hasattr(planning_problem.goal.state_list[0], 'position') is False:
            self.survival_scenario = True
        else:
            self.survival_scenario = False
            # raise AssertionError('Planning problem has no goal position (Survival Scenario).')

        # Get desired orientation
        if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):
            self.desired_orientation = self.planningProblem.goal.state_list[
                0
            ].orientation
        else:
            self.desired_orientation = Interval(-math.pi, math.pi)

        # Set lanelet costs to -1, except goal lanelet
        self.lanelet_cost = {}
        for lanelet in scenario.lanelet_network.lanelets:
            self.lanelet_cost[lanelet.lanelet_id] = -1

        # Check if there are goalLanelets_ids
        if self.goalLanelet_ids:
            for goal_lanelet_id in self.goalLanelet_ids:
                self.lanelet_cost[goal_lanelet_id] = 0

            # calculate costs for lanelets, this is a recursive method
            for goal_lanelet_id in self.goalLanelet_ids:
                visited_lanelets = []
                self.calc_lanelet_cost(
                    self.scenario.lanelet_network.find_lanelet_by_id(goal_lanelet_id),
                    1,
                    visited_lanelets,
                )

        self.print_info = print_info
        # Set the maximum length a lane change should take [m]
        self.max_lane_change_length = max_lane_change_length
        # Define path lanelets and transitions
        self.path_lanelets = []
        self.transitions = []

    def calc_lanelet_cost(
        self, cur_lanelet: Lanelet, dist: int, visited_lanelets: List[int]
    ):
        """
        Calculate distances of all lanelets which can be reached through recursive adjacency/predecessor relationship by the current lanelet.

        This is a recursive implementation.
        The calculated costs will be stored in dictionary self.lanelet_cost[Lanelet].

        Args:
            cur_lanelet (Lanelet): the current lanelet object (Often set to the goal lanelet).
            dist (int): the initial distance between 2 adjacent lanelets (Often set to 1). This value will increase recursively during the execution of this function.
            visited_lanelets (List[int]): list of visited lanelet id. In the iterations, visited lanelets will not be considered. This value changes during the recursive implementation.

        """
        if cur_lanelet.lanelet_id in visited_lanelets:
            return
        else:
            visited_lanelets.append(cur_lanelet.lanelet_id)

        if cur_lanelet.predecessor is not None:
            for pred in cur_lanelet.predecessor:
                if self.lanelet_cost[pred] == -1 or self.lanelet_cost[pred] > dist:
                    self.lanelet_cost[pred] = dist

            for pred in cur_lanelet.predecessor:
                self.calc_lanelet_cost(
                    self.lanelet_network.find_lanelet_by_id(pred),
                    dist + 1,
                    visited_lanelets,
                )

        if cur_lanelet.adj_left is not None and cur_lanelet.adj_left_same_direction:
            if (
                self.lanelet_cost[cur_lanelet.adj_left] == -1
                or self.lanelet_cost[cur_lanelet.adj_left] > dist
            ):
                self.lanelet_cost[cur_lanelet.adj_left] = dist
                self.calc_lanelet_cost(
                    self.lanelet_network.find_lanelet_by_id(cur_lanelet.adj_left),
                    dist + 1,
                    visited_lanelets,
                )

        if cur_lanelet.adj_right is not None and cur_lanelet.adj_right_same_direction:
            if (
                self.lanelet_cost[cur_lanelet.adj_right] == -1
                or self.lanelet_cost[cur_lanelet.adj_right] > dist
            ):
                self.lanelet_cost[cur_lanelet.adj_right] = dist
                self.calc_lanelet_cost(
                    self.lanelet_network.find_lanelet_by_id(cur_lanelet.adj_right),
                    dist + 1,
                    visited_lanelets,
                )

    def path_without_goal_area(self) -> np.ndarray:
        """
        Find a path if no goal area is given (survival scenario) by just following the current lanelet and all its successors.

        Returns:
            np.ndarray: Global path.

        """
        start_pos = self.planningProblem.initial_state.position
        start_orientation = self.planningProblem.initial_state.orientation

        path_lanelets = []

        # Find initial lanelet
        initial_lanelet = find_lanelet_by_position_and_orientation(
            self.lanelet_network, start_pos, start_orientation
        )[0]

        path_lanelets.append(initial_lanelet)
        successors = self.lanelet_network.find_lanelet_by_id(initial_lanelet).successor

        while successors:
            next_lanelet = successors[0]
            path_lanelets.append(next_lanelet)

            # Check if all lanelets are unique, important to avoid infinite loops
            if len(path_lanelets) > len(set(path_lanelets)):
                break

            successors = self.lanelet_network.find_lanelet_by_id(next_lanelet).successor

        # Add the initial position to the path
        path = np.array([start_pos])
        path = np.concatenate(
            (path, self.no_lane_change_first_lanelet(path_lanelets[0]))
        )

        if len(path_lanelets) > 1:
            for i in range(1, len(path_lanelets)):
                lanelet_id = path_lanelets[i]
                center_vertices = self.lanelet_network.find_lanelet_by_id(
                    lanelet_id
                ).center_vertices
                path = np.concatenate((path, center_vertices))

        return clear_duplicate_entries(path)

    def find_best_lanelet(self, lanelets: List[int]) -> int:
        """
        Find the lanelet with the lowest lanelet cost which is reachable.

        Args:
            lanelets (List[int]): IDs of the considered lanelets.

        Returns:
            int: ID of the best found lanelet.

        """
        # Check which of the final lanelets can reach the goal (>= 0, not -1) and has the best lanelet costs
        min_cost = 99999
        best_lanelet = None
        for lanelet_id in lanelets:
            # New lanelet would need better lanelet costs to replace previous best
            # Since connected_lanelets appends adjacent lanelets first, a lane change is always prioritised (right before left)
            if min_cost > self.lanelet_cost[lanelet_id] >= 0:
                min_cost = self.lanelet_cost[lanelet_id]
                best_lanelet = lanelet_id

        return best_lanelet

    def is_reached(self, position: np.array) -> bool:
        """
        Check if a given state is inside the goal region.

        Args:
            position (TYPE): Position to be checked.

        Returns:
            bool: True, if state fulfills requirements of the goal region. False if not.

        """
        # To be not to strict, allow a little bit offset (if the point would be on the edge of the given goal shape, it would not be contained)
        acceptable_offset = 0.0001

        for goal_state in self.planningProblem.goal.state_list:
            if goal_state.position.contains_point(position):
                return True
            elif goal_state.position.contains_point(
                np.array(
                    [position[0] + acceptable_offset, position[1] + acceptable_offset]
                )
            ):
                return True
            elif goal_state.position.contains_point(
                np.array(
                    [position[0] + acceptable_offset, position[1] - acceptable_offset]
                )
            ):
                return True
            elif goal_state.position.contains_point(
                np.array(
                    [position[0] - acceptable_offset, position[1] + acceptable_offset]
                )
            ):
                return True
            elif goal_state.position.contains_point(
                np.array(
                    [position[0] - acceptable_offset, position[1] - acceptable_offset]
                )
            ):
                return True

        return False

    def find_connected_lanelets(self, lanelet_id: int) -> List[int]:
        """
        Find all the lanelets, that are connected to the given lanelet and lead towards the goal (successors or adjacent in the right direction).

        Args:
            lanelet_id (int): ID of the considered lanelet.

        Returns:
            List[int]: IDs of the successor and adjacent lanelets.

        """
        lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        connected_lanelets = []
        if lanelet.adj_right is not None and lanelet.adj_right_same_direction:
            connected_lanelets.append(lanelet.adj_right)
        if lanelet.adj_left is not None and lanelet.adj_left_same_direction:
            connected_lanelets.append(lanelet.adj_left)
        # check and add successors after adjacent lanelets
        if lanelet.successor is not None:
            for lanelet_iter in lanelet.successor:
                connected_lanelets.append(lanelet_iter)

        return connected_lanelets

    def no_lane_change_first_lanelet(
        self, lanelet_id: int, start_pos: np.ndarray = None
    ) -> np.ndarray:
        """
        Find the path in a lanelet from a starting point (either initial position of planning problem or given starting point) when no lane change is necessary. Do not add all center vertices, only those after the initial position.

        Args:
            lanelet_id (int): ID of the initial lanelet.
            start_pos (np.ndarray, optional): Starting position. Defaults to None.

        Returns:
            path (np.ndarray): Path from the starting position to the end of the lanelet.

        """
        # Get the center vertices of the initial lanelet
        center_vertices = self.scenario.lanelet_network.find_lanelet_by_id(
            lanelet_id
        ).center_vertices
        # Find the best vertex to connect the initial position to
        if start_pos is None:
            start_pos = self.initial_state.position

        start_vertex = find_best_vertex(center_vertices, start_pos)
        # Add the start vertex
        path = np.array([center_vertices[start_vertex]])
        for i in range(start_vertex + 1, len(center_vertices)):
            path = np.concatenate((path, np.array([center_vertices[i]])))

        return path

    def no_lane_change_goal_lanelet(
        self, lanelet_id: int = None, vertices: np.ndarray = None
    ) -> np.ndarray:
        """
        Find the path of the goal lanelet when no lane change is necessary. Do not add all center vertices, only those before the goal is reached. Either use center vertices of given lanelet (by ID) or given vertices.

        Args:
            lanelet_id (int, optional): ID of the goal lanelet. Defaults to None.
            vertices (np.ndarray, optional): Center vertices that should be used. Defaults to None.

        Returns:
            np.ndarray: Path from the start of the lanelet to the goal.

        """
        if lanelet_id is not None:
            # Get the center vertices of the lanelet
            center_vertices = self.scenario.lanelet_network.find_lanelet_by_id(
                lanelet_id
            ).center_vertices
        else:
            center_vertices = vertices

        # Find the best vertex to connect the goal position to
        best_vertex = find_best_vertex(
            center_vertices,
            self.planningProblem.goal.state_list[0].position.center,
            position="end",
        )
        # best_vertex = len(center_vertices)-1

        # If the best vertex is the first vertex of the center vertices, the full path consists only of the goal position
        # center_vertices[0] might be already behind the goal position
        if best_vertex == 0:
            full_path = np.array(
                [self.planningProblem.goal.state_list[0].position.center]
            )
        # Else add all vertices until the best vertex
        else:
            # Initialize the full path from the start of the lanelet to the goal
            full_path = np.array([center_vertices[0]])
            # Add all vertices until the best vertex
            for i in range(1, best_vertex + 1):
                full_path = np.concatenate((full_path, np.array([center_vertices[i]])))
            # Add the goal position
            full_path = np.concatenate(
                (
                    full_path,
                    np.array([self.planningProblem.goal.state_list[0].position.center]),
                )
            )

        # Only return the vertices until one of them reached the goal
        return self.add_path_as_long_as_in_goal(full_path)
        # return full_path

    def start_and_goal_in_one_lanelet(
        self, lanelet_id: int, start_pos: np.ndarray = None
    ) -> np.ndarray:
        """
        Find the path if the start (either initial position of the planning problem or given starting position) and the goal are in the same lanelet.

        Args:
            lanelet_id (int): ID of the lanelet.
            start_pos (np.ndarray, optional): Start position. Defaults to None.

        Returns:
            path (np.ndarray): Path from the initial position to the goal position.

        """
        # Find the vertices from the initial position to the end of the lanelet
        vertices_from_start = self.no_lane_change_first_lanelet(
            lanelet_id, start_pos=start_pos
        )

        # Only return the vertices before reaching the goal
        # return self.no_lane_change_goal_lanelet(vertices=vertices_from_start)
        return vertices_from_start

    def add_path_as_long_as_in_goal(self, path: np.ndarray) -> np.ndarray:
        """
        Find the vertices of a path as long as they are in the goal area.

        Args:
            path (np.ndarray): Path to be examined.

        Returns:
            path_until_goal (TYPE): Path until goal is leaved again.

        """
        goal_reached_once = False
        for i in range(len(path)):
            # Break if one point reaches the goal
            if goal_reached_once is False and self.is_reached(path[i]):
                goal_reached_once = True

            if goal_reached_once is True and self.is_reached(path[i]) is False:
                break

            if i == 0:
                path_until_goal = np.array([path[0]])
            else:
                path_until_goal = np.concatenate((path_until_goal, np.array([path[i]])))

        return path_until_goal

    def get_goal_orientation(self) -> float:
        """
        Calculate the orientation at the goal position.

        Returns:
            float: Orientation at the goal position.

        """
        # Get the center vertices
        end_center_vertices = self.scenario.lanelet_network.find_lanelet_by_id(
            self.path_lanelets[-1]
        ).center_vertices
        # Calculate the range of the interval
        interval_range = abs(
            self.desired_orientation.start - self.desired_orientation.end
        )
        # If the interval is greater than 45° find the vertex closest to the goal position and calculate the orientation of the center line at that point
        if interval_range >= np.deg2rad(45):
            closest_vertex = find_closest_vertex(
                end_center_vertices,
                self.planningProblem.goal.state_list[0].position.center,
            )
            return calc_angle_of_position(
                end_center_vertices, end_center_vertices[closest_vertex]
            )
        # If the range is not that big, chose the mean of the interval
        else:
            return self.desired_orientation.start + interval_range / 2

    def find_new_endpoint(
        self,
        path_lanelet_index_start: int,
        center_vertices: np.ndarray,
        n_consecutive_lane_changes: int = 1,
    ):
        """
        Find a new endpoint on the center line defined by the center vertices, that enables a faster lane change.

        Args:
            path_lanelet_index_start (int): Position of the lanelet where the lane change start in the path_lanelets array.
            center_vertices (np.ndarray): Center line of an array that holds all the possible endpoints.
            n_consecutive_lane_changes (int, optional): Number of consecutive lane changes to reach the lanelet. Defaults to 1.

        Returns:
            end_point (np.ndarray): New endpoint.
            end_orientation (float): Orientation of the new endpoint.

        """
        length = 0
        vertex_id = 0
        # Loop through the vertices until the length is to long
        # Result in the id of the vertex at which the max lane change distance is exceeded
        while length < n_consecutive_lane_changes * self.max_lane_change_length:
            # Check if we are in the first iteration which equals that we start from self.initial_state.position
            # Do not add up all the length starting from the first vertex because the initial position might be already half way through the lanelet
            # Only in the first step (when length == 0)
            if path_lanelet_index_start == 0 and length == 0:
                # Find the point on the center line of the lanelet, that is closest to the initial position
                point_on_line, next_index = calc_point_on_line_closest_to_another_point(
                    center_vertices, self.initial_state.position
                )
                # Add the length from this initial position to the next center line vertex
                length = length + euclidean_distance(
                    point_on_line, center_vertices[next_index]
                )
                # Safe the id of the center vertex
                vertex_id = next_index
                continue
            # If we do not start from the initial position, we can start adding up the lengths from the beginning of the lanelet
            elif vertex_id < len(center_vertices) - 1:
                length = length + euclidean_distance(
                    center_vertices[vertex_id], center_vertices[vertex_id + 1]
                )
            # If there would be an indexing error, return the last point of the vertices
            # Should only occur, when the actual polyline of the initial lane change is longer than accepted, but aligned to the lanelet center line, it is not longer than accepted.
            else:
                orientation = calc_orientation_of_line(
                    center_vertices[vertex_id], center_vertices[vertex_id - 1]
                )
                return center_vertices[-1], orientation + math.pi

            # Increment the vertex id
            vertex_id += 1

        # Go back from the vertex the lane change distance was exceeded until the distance is ok
        # Get the orientation of the line from the vertex exceeding the limit to the previous vertex
        orientation = calc_orientation_of_line(
            center_vertices[vertex_id], center_vertices[vertex_id - 1]
        )
        # Calculate the distance the limit was exceeded
        reverse = length - n_consecutive_lane_changes * self.max_lane_change_length
        # Get the new end point by subtracting the exceeded distance in direction of the center line from the vertex
        end_point_x = center_vertices[vertex_id][0] + reverse * math.cos(orientation)
        end_point_y = center_vertices[vertex_id][1] + reverse * math.sin(orientation)
        end_point = np.array([end_point_x, end_point_y])

        # end_orientation is the reversed orientation
        end_orientation = orientation + math.pi

        return end_point, end_orientation

    def get_polyline_endpoint(
        self,
        path_lanelet_index_end: int,
        start_point: np.ndarray,
        path_lanelet_index_start: int,
        n_consecutive_lane_changes: int = 1,
    ):
        """
        Find an endpoint for a lane change that has a length not longer than the give parameter.

        Args:
            path_lanelet_index_end (int): On which position is the lanelet where the lane change should end in the path_lanelets array.
            start_point (np.ndarray): Starting point of the lane change.
            path_lanelet_index_start (int): On which position is the lanelet where the lane change should start in the path_lanelets array.
            n_consecutive_lane_changes (int, optional): How many lanes are changed in this transition. Defaults to 1.

        Returns:
            end_point (np.ndarray): End point, where the lane change should end.
            end_orientation (float): Desired orientation of the path at the end of the lane change.
            initially_too_long (bool): Marker, if the initial lane change took to long. If so, the points in the lanelet after the lane change is finished need to be added to the path later.

        """
        # Get the center vertices of the lanelet containing the end point
        end_center_vertices = self.scenario.lanelet_network.find_lanelet_by_id(
            self.path_lanelets[path_lanelet_index_end]
        ).center_vertices

        # Marker if the initial lane change was to long
        # The points from the end of the polyline to the goal/end of the lanelet need to be added later on
        initially_too_long = False

        # If the end lanelet is not the goal lanelet, end at the last vertex of the end lanelet
        if path_lanelet_index_end < len(self.path_lanelets) - 1:
            # The last center vertex of the end lanelet is the end point
            end_point = end_center_vertices[-1]
            end_orientation = calc_orientation_of_line(
                end_center_vertices[-2], end_center_vertices[-1]
            )
        # Else end at the goal position
        else:
            # If the end lanelet is the goal lanelet, end at the goal position
            end_point = self.planningProblem.goal.state_list[0].position.center
            end_orientation = self.get_goal_orientation()

        # Check if the lane change is too long and if yes, get a new end point
        if (
            euclidean_distance(start_point, end_point)
            > n_consecutive_lane_changes * self.max_lane_change_length
        ):
            # Set marker
            initially_too_long = True
            end_point, end_orientation = self.find_new_endpoint(
                path_lanelet_index_start,
                end_center_vertices,
                n_consecutive_lane_changes=n_consecutive_lane_changes,
            )

        return end_point, end_orientation, initially_too_long

    def get_consecutive_lane_changes(self, transition_index: int) -> int:
        """
        Get the number of consecutive lane changes for a transition.

        Args:
            transition_index (int): Index of the considered transition in the transition array.

        Returns:
            consecutive_lane_changes (int): Number of consecutive lane changes.

        """
        consecutive_lane_changes = 0
        # Loop the transition array from the given transition on
        while transition_index < len(self.transitions):
            if self.transitions[transition_index] == "change":
                transition_index += 1
                consecutive_lane_changes += 1
            else:
                break

        return consecutive_lane_changes

    def split_global_path(self, global_path: np.ndarray):
        """
        Split the global path the part until it reaches the goal and the part after leaving the goal.

        Args:
            global_path (np.ndarray): Global path.

        Returns:
            np.ndarray: Path until the goal is left again.
            np.ndarray: Path after the goal is left again.

        """
        if self.survival_scenario is True:
            return global_path, np.concatenate(([global_path[-2]], [global_path[-1]]))
        else:
            reached = False
            split_index = len(global_path)
            for index, point in enumerate(global_path):
                if self.is_reached(position=point):
                    if reached is False:
                        reached = True
                else:
                    if reached is True:
                        split_index = index
                        break
            split_path = np.split(global_path, [split_index])
            to_goal_path = split_path[0]
            after_goal = np.concatenate(([to_goal_path[-1]], split_path[1]))

            return to_goal_path, after_goal

    def plan_global_path(self):
        """
        Find a path that solves the planning problem.

        Returns:
            np.ndarray: Path connecting the initial position and the goal position.
            float: Path length.

        """
        # Lanelet based search algorithm
        if self.print_info:
            print("Searching for global path...")

        # If it is a survival scenario without goal area, just follow the start lanelet and all its successors
        if self.survival_scenario is True:
            path = self.path_without_goal_area()
            return path, calc_path_length(path)

        # Find the id of the best lanelet of the initial position
        start_lanelet_id = self.find_best_lanelet(self.startLanelet_ids)

        # If the goal position can not be reached, return None
        if start_lanelet_id is None:
            return None

        # Add the initial position to the path
        path = np.array([self.initial_state.position])

        # Find the lanelets that connect the initial position and the goal position and the necessary transitions (stay on lane or change lane)
        self.path_lanelets.append(start_lanelet_id)

        # Check if the first lanelet has another lanelet to append
        if self.find_best_lanelet(self.find_connected_lanelets(self.path_lanelets[-1])):
            lanelet_available = True
        else:
            lanelet_available = False

        # Append while there are connected and better or equal lanelets (does not need to be set to False, since it will break)
        while lanelet_available is True:
            connected_lanelets = self.find_connected_lanelets(self.path_lanelets[-1])
            best_lanelet = self.find_best_lanelet(connected_lanelets)
            # If there is a lanelet be connected, this lanelet is better or equal and the lanelet was not visited yet
            if (
                best_lanelet
                and self.lanelet_cost[best_lanelet]
                <= self.lanelet_cost[self.path_lanelets[-1]]
                and best_lanelet not in self.path_lanelets
            ):
                self.path_lanelets.append(best_lanelet)
                lanelet_available = True
                # Check if the lane was changes or it stays in the lane
                if (
                    best_lanelet
                    in self.scenario.lanelet_network.find_lanelet_by_id(
                        self.path_lanelets[-2]
                    ).successor
                ):
                    self.transitions.append("stay")
                else:
                    self.transitions.append("change")
            else:
                break

        # If the goal position is given in form of lanelets (converted to shape groups later on), set the goal position to the end vertex of the goal lanelet
        # If the goal position is given as a position, the center of this position stays the goal position
        if hasattr(self.planningProblem.goal.state_list[0].position, 'center') is False:
            self.planningProblem.goal.state_list[
                0
            ].position.center = self.scenario.lanelet_network.find_lanelet_by_id(
                self.path_lanelets[-1]
            ).center_vertices[
                -1
            ]
        # self.planningProblem.goal.state_list[0].position.center = self.scenario.lanelet_network.find_lanelet_by_id(self.path_lanelets[-1]).center_vertices[-1]

        # print("path lanelets: ", self.path_lanelets)
        # print("transitions: ", self.transitions)

        # Create the iterator for the path lanelets
        path_lanelet_indices = range(0, len(self.path_lanelets))
        path_lanelet_iter = iter(path_lanelet_indices)

        for path_lanelet_index in path_lanelet_iter:

            # Check if it is the last lanelet (no more transition)
            if path_lanelet_index == len(self.transitions):
                # Can not be a lane change because it would skip this iteration

                # Check if there is no transition at all:
                if len(self.transitions) == 0:
                    path = np.concatenate(
                        (
                            path,
                            self.start_and_goal_in_one_lanelet(self.path_lanelets[0]),
                        )
                    )
                # If there have been transitions before, only add the path to the goal
                else:
                    # path = np.concatenate((path, self.no_lane_change_goal_lanelet(lanelet_id=self.path_lanelets[path_lanelet_index])))
                    path = np.concatenate(
                        (
                            path,
                            self.scenario.lanelet_network.find_lanelet_by_id(
                                self.path_lanelets[path_lanelet_index]
                            ).center_vertices,
                        )
                    )

            # If there is a transition
            else:
                # No lane change
                if self.transitions[path_lanelet_index] == "stay":
                    # Add center vertices of the start lanelet from initial position on
                    if path_lanelet_index == 0:
                        # Add vertices of the start lanelet, but only those after the initial position
                        path = np.concatenate(
                            (
                                path,
                                self.no_lane_change_first_lanelet(
                                    self.path_lanelets[0]
                                ),
                            )
                        )

                    # Add all vertices of a lanelet, if it is not the first lanelet
                    else:
                        path = np.concatenate(
                            (
                                path,
                                self.scenario.lanelet_network.find_lanelet_by_id(
                                    self.path_lanelets[path_lanelet_index]
                                ).center_vertices,
                            )
                        )

                # Lane change
                else:
                    # Get the vertices of the start and end lanelet of the lane change
                    start_center_vertices = (
                        self.scenario.lanelet_network.find_lanelet_by_id(
                            self.path_lanelets[path_lanelet_index]
                        ).center_vertices
                    )

                    # Get number of consecutive lane changes
                    consecutive_lane_changes = self.get_consecutive_lane_changes(
                        path_lanelet_index
                    )
                    path_lanelet_index_end = (
                        path_lanelet_index + consecutive_lane_changes
                    )

                    # If it is the first lanelet, start from the initial position
                    if path_lanelet_index == 0:
                        start_point = self.initial_state.position
                        start_orientation = self.initial_state.orientation
                    # Else start from the first vertex of the start lanelet
                    else:
                        start_point = start_center_vertices[0]
                        start_orientation = calc_orientation_of_line(
                            start_center_vertices[0], start_center_vertices[1]
                        )

                    # Get the end point for the lane change
                    (
                        end_point,
                        end_orientation,
                        initially_too_long,
                    ) = self.get_polyline_endpoint(
                        path_lanelet_index_end,
                        start_point,
                        path_lanelet_index,
                        n_consecutive_lane_changes=consecutive_lane_changes,
                    )

                    # Get the path of the lane change
                    lane_change_path = quintic_polynomials_planner(
                        start_point, start_orientation, end_point, end_orientation
                    )
                    path = np.concatenate((path, lane_change_path))

                    # If the initial lane change was too long, the vertices after the lane change was done need to be added
                    if initially_too_long:
                        # The lane change ends in the last lanelet
                        if path_lanelet_index_end == len(self.path_lanelets) - 1:
                            # Add the path from the end of the lane change to the goal position
                            path = np.concatenate(
                                (
                                    path,
                                    self.start_and_goal_in_one_lanelet(
                                        self.path_lanelets[path_lanelet_index_end],
                                        end_point,
                                    ),
                                )
                            )
                        # The lane change does not end in the goal lanelet
                        else:
                            # Add the path from the end of the lane change to the last vertex of the lanelet
                            path = np.concatenate(
                                (
                                    path,
                                    self.no_lane_change_first_lanelet(
                                        self.path_lanelets[path_lanelet_index_end],
                                        end_point,
                                    ),
                                )
                            )

                    # Skip following lanelets after a lane change since a lane change covers at least two lanelets
                    for next_id in range(consecutive_lane_changes):
                        next(path_lanelet_iter)

        # Add successors after the goal lanelet
        if self.scenario.lanelet_network.find_lanelet_by_id(
            self.path_lanelets[-1]
        ).successor:
            path_lanelets_after_goal = [
                self.scenario.lanelet_network.find_lanelet_by_id(
                    self.path_lanelets[-1]
                ).successor[0]
            ]
            successors_after_the_goal_available = True

            while successors_after_the_goal_available is True:
                path = np.concatenate(
                    (
                        path,
                        self.scenario.lanelet_network.find_lanelet_by_id(
                            path_lanelets_after_goal[-1]
                        ).center_vertices,
                    )
                )
                successors_after_the_goal_available = False
                if self.scenario.lanelet_network.find_lanelet_by_id(
                    path_lanelets_after_goal[-1]
                ).successor:
                    successors_after_the_goal_available = True
                    path_lanelets_after_goal.append(
                        self.scenario.lanelet_network.find_lanelet_by_id(
                            path_lanelets_after_goal[-1]
                        ).successor[0]
                    )

        if self.print_info:
            print("Goal reached!!")
            print(
                "Number of visited lanelets: %d, Number of lane changes: %d"
                % (len(self.path_lanelets), self.transitions.count("change"))
            )

        # Clear duplicate entries from the path
        clean_path = clear_duplicate_entries(path)

        # Calculate the path length
        path_length = calc_path_length(clean_path)

        return clean_path, path_length


def clear_duplicate_entries(path: np.ndarray) -> np.ndarray:
    """
    Clear all the duplicates from a path.

    Args:
        path (np.ndarray): Path that should be freed from duplicates.

    Returns:
        np.ndarray: Path free of duplicates.

    """
    # Find the duplicate entries
    new_array = [tuple(row) for row in path]
    # Find the indices (sorted my the x-value) of the unique entries
    _, indices = np.unique(new_array, return_index=True, axis=0)

    # return the entries of path on the unique indices (sorted by index)
    return path[np.sort(indices)]


def find_closest_vertex(center_vertices: np.ndarray, pos: np.ndarray) -> int:
    """
    Find the index of the closest center vertices to the given position.

    Args:
        center_vertices (np.ndarray): Center vertices.
        pos (np.ndarray): Position that should be checked.

    Returns:
        int: Index of the center vertex, that is closest to the given position.

    """
    distances = []
    for vertex in center_vertices:
        distances.append(euclidean_distance(vertex, pos))
    return distances.index(min(distances))


def calc_angle_of_position(center_vertices: np.ndarray, pos: np.ndarray) -> float:
    """
    Return the angle (in world coordinate, radian) of the line defined by 2 nearest lanelet center vertices to the given position.

    Args:
        center_vertices (np.ndarray): Lanelet center vertices, whose distance to the given position is considered..
        pos (np.ndarray): The position to be considered..

    Returns:
        float: Angle of the line defined by two nearest lanelet center vertices to the given position.

    """
    index_closest_vert = find_closest_vertex(center_vertices, pos)
    if index_closest_vert + 1 >= center_vertices.size / 2.0:
        index_closest_vert = index_closest_vert - 1
    return calc_orientation_of_line(
        center_vertices[index_closest_vert], center_vertices[index_closest_vert + 1]
    )


def euclidean_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate the euclidean distance between two points.

    Args:
        pos1 (np.ndarray): Point 1.
        pos2 (np.ndarray): Point 2.

    Returns:
        float: Euclidean distance between the two given points.

    """
    return np.sqrt(
        (pos1[0] - pos2[0]) * (pos1[0] - pos2[0])
        + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])
    )


def orientation_diff(orientation_1: float, orientation_2: float) -> float:
    """
    Calculate the orientation difference between two orientations in radians.

    Args:
        orientation_1 (float): Orientation 1.
        orientation_2 (float): Orientation 2.

    Returns:
        float: Orientation difference in radians.

    """
    return math.pi - abs(abs(orientation_1 - orientation_2) - math.pi)


def calc_orientation_of_line(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the orientation of the line connecting two points (angle in radian, counter-clockwise defined).

    Args:
        point1 (np.ndarray): Starting point.
        point2 (np.ndarray): Ending point.

    Returns:
        float: Orientation in radians.

    """
    return math.atan2(point2[1] - point1[1], point2[0] - point1[0])


def find_best_vertex(
    center_vertices: np.ndarray,
    point_to_be_connected: np.ndarray,
    position: str = "start",
) -> int:
    """
    Find the index of the vertex that connects a given point the best.

    Args:
        center_vertices (np.ndarray): Center vertices in which the given point should be merged.
        point_to_be_connected (np.ndarray): Point that should be merged in the center vertices.
        position (str, optional): Should the point be merged as an end point or as a start point. Defaults to "start".

    Returns:
        int: Index of the vertex tat integrates the given point the best.

    """
    # Find closest vertex
    closest_vertex = find_closest_vertex(center_vertices, point_to_be_connected)

    # If the point is a starting point
    if position == "start":
        if closest_vertex == 0:
            start_vertex = 1
        elif closest_vertex == len(center_vertices) - 1:
            start_vertex = len(center_vertices) - 1
        else:
            for i in range(closest_vertex, closest_vertex + 2):
                orientation_of_lanelet = calc_orientation_of_line(
                    center_vertices[i - 1], center_vertices[i]
                )
                orientation_of_connection = calc_orientation_of_line(
                    point_to_be_connected, center_vertices[i]
                )
                orientation_difference = orientation_diff(
                    orientation_of_lanelet, orientation_of_connection
                )
                if orientation_difference <= math.pi / 2:
                    return i
        return start_vertex
    # If the point is an ending point
    elif position == "end":
        if closest_vertex == 0:
            end_vertex = 0
        elif closest_vertex == len(center_vertices) - 1:
            end_vertex = len(center_vertices) - 2
        else:
            for i in range(closest_vertex, closest_vertex + 2):
                orientation_of_lanelet = calc_orientation_of_line(
                    center_vertices[i - 1], center_vertices[i]
                )
                orientation_of_connection = calc_orientation_of_line(
                    center_vertices[i - 1], point_to_be_connected
                )
                orientation_difference = orientation_diff(
                    orientation_of_lanelet, orientation_of_connection
                )
                if orientation_difference <= math.pi / 2:
                    end_vertex = i - 1
        return end_vertex


def calc_point_on_line_closest_to_another_point(
    center_vertices: np.ndarray, p3: np.ndarray
):
    """
    Given a point and the center vertices of a lanelet, find the point on the center line, that is closest to the given point.

    Args:
        center_vertices (np.ndarray): Center vertices of a lanelet.
        p3 (np.ndarray): Point for which the closest point on the center line is searched.

    Returns:
        np.ndarray: Point on the center line with the smallest distance to p3.
        int: Position of the vertex of the center line right behind the returned point.

    """
    index_closest_vert = find_closest_vertex(center_vertices, p3)
    if index_closest_vert + 1 >= center_vertices.size / 2.0:
        index_closest_vert = index_closest_vert - 1
    p1 = center_vertices[index_closest_vert]
    p2 = center_vertices[index_closest_vert + 1]

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    a = (dy * (y3 - y1) + dx * (x3 - x1)) / det

    return np.array([x1 + a * dx, y1 + a * dy]), index_closest_vert + 1


def calc_path_length(path: np.ndarray) -> float:
    """
    Return the travelled distance of the given path.

    Args:
        path (np.ndarray): The path, whose travelled euclidean distance is calculated.

    Returns:
        float: Length of the path.

    """
    dist = 0
    for i in range(len(path) - 1):
        dist = dist + euclidean_distance(path[i], path[i + 1])
    return dist


def find_lanelet_by_position_and_orientation(lanelet_network, position, orientation):
    """Return the IDs of lanelets within a certain radius calculated from an initial state (position and orientation).

    Args:
        lanelet_network ([CommonRoad LaneletNetwork Object]): [description]
        position ([np.array]): [position of the vehicle to find lanelet for]
        orientation ([type]): [orientation of the vehicle for finding best lanelet]

    Returns:
        [int]: [list of matching lanelet ids]
    """
    # TODO: Shift this function to commonroad helpers
    lanelets = []
    initial_lanelets = lanelet_network.find_lanelet_by_position([position])[0]
    best_lanelet = initial_lanelets[0]
    radius = math.pi / 5.0  # ~0.63 rad = 36 degrees, determined empirically
    min_orient_diff = math.inf
    for lnlet in initial_lanelets:
        center_line = lanelet_network.find_lanelet_by_id(lnlet).center_vertices
        lanelet_orientation = calc_orientation_of_line(center_line[0], center_line[-1])
        orient_diff = orientation_diff(orientation, lanelet_orientation)

        if orient_diff < min_orient_diff:
            min_orient_diff = orient_diff
            best_lanelet = lnlet
            if orient_diff < radius:
                lanelets = [lnlet] + lanelets
        elif orient_diff < radius:
            lanelets.append(lnlet)

    if not lanelets:
        lanelets.append(best_lanelet)

    return lanelets


if __name__ == '__main__':

    # Change the working directory to the directory of the evaluation script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Read scenario
    print("Loading scenario...")

    # Option 1: enter one specific file
    # get path of the scenario

    scenario_name = "recorded/scenario-factory/DEU_Speyer-4_2_T-1.xml"
    # scenario_name = 'NGSIM/Peachtree/USA_Peach-1_1_T-1.xml'
    # scenario_name = 'NGSIM/US101/USA_US101-16_2_T-1.xml'
    # scenario_name = 'hand-crafted/DEU_Muc-2_1_T-1.xml'
    # scenario_name = 'SUMO/ESP_Mad-2_1_T-1.xml'
    # scenario_name = 'THI-Bicycle/RUS_Bicycle-4_1_T-1.xml'

    scenario_path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        ),
        'commonroad-scenarios/scenarios',
        scenario_name,
    )

    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()

    # Plot lanelet network and planning problem
    # draw_object(scenario.lanelet_network)
    # draw_object(planning_problem_set)
    # plt.gca().set_aspect('equal')
    # plt.show()

    # Get planning problem
    # Take the first planning problem, if the scenario consists of multiple planning problems
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # Initialize path planner
    path_planner = LaneletPathPlanner(scenario, planning_problem)

    start_time = time.time()
    # Execute the search
    path, path_length = path_planner.plan_global_path()

    # Get the execution time
    exec_time = time.time() - start_time

    print('Global path planning took {0:.3f} seconds.'.format(exec_time))

    # print("path: \n", result)

    # Plot the resulting trajectory
    print("Resulting global path with a length of %.2f meter: " % path_length)
    draw_object(scenario.lanelet_network)
    draw_object(planning_problem)
    path_to_goal, path_after_goal = path_planner.split_global_path(global_path=path)
    plt.plot(path_to_goal[:, 0], path_to_goal[:, 1], color='red', zorder=20)
    plt.plot(
        path_after_goal[:, 0],
        path_after_goal[:, 1],
        color='red',
        zorder=20,
        linestyle='--',
    )
    # plt.plot(path[:, 0], path[:, 1], color='red', zorder=20)
    plt.gca().set_aspect('equal')
    plt.show()
