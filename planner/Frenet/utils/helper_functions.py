#!/user/bin/env python

"""Helper functions for the frenét planner."""

import math
import os
import enum
from pathlib import Path

import commonroad_dc.pycrcc as pycrcc
import matplotlib.colors as colors
import numpy as np
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_object,
)
from commonroad_dc.pycrcc import ShapeGroup
import shapely
from commonroad_helper_functions.sensor_model import (
    __unit_vector,
    _calc_corner_points,
    _create_polygon_from_vertices,
    _identify_projection_points,
)
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from numba import njit
from typing import Any, List


def print_planning_problem(
    planning_problem: PlanningProblem, lanelet_network: LaneletNetwork
):
    """
    Print infos about the planning problem.

    Args:
        planning_problem (PlanningProblem): Considered planning problem.
        lanelet_network: Considered lanelet network.
    """
    print("Number of planning problems: %d" % len(planning_problem.goal.state_list))
    for i in range(len(planning_problem.goal.state_list)):

        # get center positions of the goal areas
        if (
            hasattr(planning_problem.goal, "lanelets_of_goal_position")
            and planning_problem.goal.lanelets_of_goal_position is not None
        ):
            goal_lanelet_id = planning_problem.goal.lanelets_of_goal_position[i]
            goal_lanelet_obj = lanelet_network.find_lanelet_by_id(goal_lanelet_id[0])
            target_x = goal_lanelet_obj.center_vertices[-1][0]
            target_y = goal_lanelet_obj.center_vertices[-1][1]
        elif hasattr(planning_problem.goal.state_list[i], "position"):
            if hasattr(planning_problem.goal.state_list[i].position, "center"):
                target_x = planning_problem.goal.state_list[i].position.center[0]
                target_y = planning_problem.goal.state_list[i].position.center[1]
        else:
            target_x = None
            target_y = None

        # get goal orientation
        if hasattr(planning_problem.goal.state_list[i], "orientation"):
            target_orientation_interval = planning_problem.goal.state_list[
                i
            ].orientation
            target_orientation_start = target_orientation_interval.start
            target_orientation_end = target_orientation_interval.end
        else:
            target_orientation_start = 99.9
            target_orientation_end = 99.9

        # get goal velocity
        if hasattr(planning_problem.goal.state_list[i], "velocity"):
            target_velocity_interval = planning_problem.goal.state_list[i].velocity
            target_velocity_start = target_velocity_interval.start
            target_velocity_end = target_velocity_interval.end
        else:
            target_velocity_start = 99.99
            target_velocity_end = 99.99

        # get goal time
        if hasattr(planning_problem.goal.state_list[i], "time_step"):
            target_time_interval = planning_problem.goal.state_list[i].time_step
            target_time_step_start = target_time_interval.start
            target_time_step_end = target_time_interval.end
        else:
            target_time_step_start = 99
            target_time_step_end = 99

        print(
            "Planning problem %d:\nTarget-position: [%.2f, %.2f]\nTarget-velocity: %.2f - %.2f m/s\nTarget-orientation: %.2f - %.2f rad\nTarget-time-step: %d - %d"
            % (
                planning_problem.planning_problem_id,
                target_x,
                target_y,
                target_velocity_start,
                target_velocity_end,
                target_orientation_start,
                target_orientation_end,
                target_time_step_start,
                target_time_step_end,
            )
        )


@njit
def distance(pos1: np.array, pos2: np.array):
    """
    Return the euclidean distance between 2 points.

    Args:
        pos1 (np.array): First point.
        pos2 (np.array): Second point.

    Returns:
        float: Distance between point 1 and point 2.
    """
    return np.linalg.norm(pos1 - pos2)


def get_max_curvature(vehicle_params, v: float = 0.0):
    """
    Get the maximum drivable curvature for a given vehicle and velocity.

    Args:
        vehicle_params (VehicleParameters): Parameters of the considered vehicle.
        v (float): Driven velocity. Defaults to 0.0.

    Returns:
        float: Maximum curvature.
        str: Velocity mode.

    """
    # calculate the turning radius
    turning_radius = math.sqrt(
        (vehicle_params.l ** 2 / math.tan(vehicle_params.steering.max) ** 2)
        + (vehicle_params.l_r ** 2)
    )
    # below this velocity, it is assumed that the vehicle can follow the turning radius
    # above this velocity, the curvature is calculated differently
    threshold_low_velocity = np.sqrt(vehicle_params.lateral_a_max * turning_radius)

    # get curvature via turning radius
    if v < threshold_low_velocity:
        return 1.0 / turning_radius, "lvc"
    # get velocity dependent curvature (curvature derived from the lateral acceleration)
    else:
        c_max_current = vehicle_params.lateral_a_max / (v ** 2)
        return c_max_current, "hvc"


def logistic_function(x: float, x_0: float = 40.0, k: float = 0.1465):
    """
    Return the y-value of a logistic function with the given parameters.

    Args:
        x: Considered x-value.
        x_0: x_0 parameter for the center of the logistic function. Defaults to 40.
        k: k parameter for the slope of the logistic function. Defaults to 0.1465.

    Returns:
        float: y-value of the logistic function at x.
    """
    # equation of a logistic function
    return 1 / (1 + np.exp(-k * (x - x_0)))


def get_harm_distribution():
    """
    Create and print a matrix for the harm distribution of all vehicle types available in the commonroad environment.

    This matrix is then used for the risk assessment.
    """
    # set the safety factors of the different vehicles
    safety_factors = [8.0, 12.0, 12.0, 1.0, 1.0, 8.0, 8.0, 12, 2]

    # create the matrix for the harm distribution
    harm_dist_matrix = []

    # calculate the harm distribution
    # the safer the vehicle, the less harm it takes --> harm_dist_vehicle_1 = (safety_factor_vehicle_2 (safety_factor_vehicle_1 + safety_factor_vehicle_2))
    for ego in range(len(safety_factors)):
        harm_dist_row = []
        for obst in range(len(safety_factors)):
            harm_dist = safety_factors[obst] / (
                safety_factors[ego] + safety_factors[obst]
            )
            harm_dist_row.append(round(harm_dist, 2))
        harm_dist_matrix.append(harm_dist_row)

    print(harm_dist_matrix)


def create_tvobstacle(
    traj_list: [[float]], box_length: float, box_width: float, start_time_step: int
):
    """
    Return a time variant collision object.

    Args:
        traj_list ([[float]]): List with the trajectory ([x-position, y-position, orientation]).
        box_length (float): Length of the obstacle.
        box_width (float): Width of the obstacle.
        start_time_step (int): Time step of the initial state.

    Returns:
        pyrcc.TimeVariantCollisionObject: Collision object.
    """
    # time variant object starts at the given time step
    tv_obstacle = pycrcc.TimeVariantCollisionObject(time_start_idx=start_time_step)
    for state in traj_list:
        # append each state to the time variant collision object
        tv_obstacle.append_obstacle(
            pycrcc.RectOBB(box_length, box_width, state[2], state[0], state[1])
        )
    return tv_obstacle


def get_goal_area_shape_group(planning_problem, scenario):
    """
    Return a shape group that represents the goal area.

    Args:
        planning_problem (PlanningProblem): Considered planning problem.
        scenario (Scenario): Considered scenario.

    Returns:
        ShapeGroup: Shape group representing the goal area.
    """
    # get goal area collision object
    # the goal area is either given as lanelets
    if (
        hasattr(planning_problem.goal, "lanelets_of_goal_position")
        and planning_problem.goal.lanelets_of_goal_position is not None
    ):
        # get the polygons of every lanelet
        lanelets = []
        for lanelet_id in planning_problem.goal.lanelets_of_goal_position[0]:
            lanelets.append(
                scenario.lanelet_network.find_lanelet_by_id(
                    lanelet_id
                ).convert_to_polygon()
            )

        # create a collision object from these polygons
        goal_area_polygons = create_collision_object(lanelets)
        goal_area_co = ShapeGroup()
        for polygon in goal_area_polygons:
            goal_area_co.add_shape(polygon)

    # or the goal area is given as positions
    elif hasattr(planning_problem.goal.state_list[0], "position"):
        # get the polygons of every goal area
        goal_areas = []
        for goal_state in planning_problem.goal.state_list:
            goal_areas.append(goal_state.position)

        # create a collision object for these polygons
        goal_area_polygons = create_collision_object(goal_areas)
        goal_area_co = ShapeGroup()
        for polygon in goal_area_polygons:
            goal_area_co.add_shape(polygon)

    # or it is a survival scenario
    else:
        goal_area_co = None

    return goal_area_co


def create_log_file(
    directory_path: str,
    log_msgs: [str],
    n_scenarios: int,
    success_rate: float,
    success_rate_strict_time: float,
    exec_time: float,
    avg_exec_time: float,
    avg_suc_exec_time: float,
    failures_dict: dict,
):
    """
    Create a log file for the evaluation results.

    Args:
        directory_path (str): Directory path for the log file.
        log_msgs ([str]): Log messages.
        n_scenarios (int): Number of scenarios.
        success_rate (float): Success rate for all scenarios.
        success_rate_strict_time (float): Success rate for all scenarios with strict time.
        exec_time (float): Total execution time.
        avg_exec_time (float): Average execution time for a scenario.
        avg_suc_exec_time (float): Average execution time for a successful scenario.
        failures_dict (dict): Dictionary containing all the resons why the planner fails an how often they occur.
    """
    # open the log file
    log_file = open(os.path.join(directory_path, "result_logs.txt"), "w")

    # write the labels of the columns
    log_file.write("Benchmark ID; Succeeded; Execution Time; Fail-Message\n")
    # write information about every scenario
    for msg in log_msgs:
        log_file.write(msg)

    # get some information about the success rate
    n_success = int((n_scenarios * success_rate) / 100.0)
    n_failure = n_scenarios - n_success

    log_file.write(
        "\nNumber of evaluated scenarios: %d\nSuccesses: %d, Failures: %d\nSuccess-rate: %.2f %%\nSuccess-rate with strict time: %.2f %%\nTotal execution time: %.2f s, average execution time: %.2f s, avg successful execution time: %.2f s"
        % (
            n_scenarios,
            n_success,
            n_failure,
            success_rate,
            success_rate_strict_time,
            exec_time,
            avg_exec_time,
            avg_suc_exec_time,
        )
    )

    # get some information why the failed scenarios fail
    log_file.write("\n\nResons for failure:\n")
    for item in failures_dict:
        log_file.write(item[0] + ": " + str(item[1]) + "\n")

    # close the log file
    log_file.close()


def save_solution_trajectory(
    solution_trajectory,
    vehicle_params,
    success: bool,
    output_path: str,
    ego_id: int,
    scenario,
    planning_problem_set,
):
    """
    Save a xml file with the solution of the given planning problem.

    Args:
        solution_trajectory (StateList): The solution trajectory.
        vehicle_params (VehicleParameters): The parameters of the vehicle used to solve the problem.
        success (bool): True if the scenario was solved successfully.
        output_path (str): Path where the solution file should be stored.
        ego_id (int): ID of the ego vehicle.
        scenario (Scenario): Considered scenario.
        planning_problem_set (PlanningProblemSet): Considered planning problem set.
    """
    # create the directory structure (successfully solved scenarios are stored in different directories than failed ones)
    if success is True:
        path_add_on = "success/"
    else:
        path_add_on = "failure/"

    # get the path where the solution file should be saved
    path = output_path + path_add_on

    # create path for solutions
    Path(path).mkdir(parents=True, exist_ok=True)

    # create the trajectory of the obstacle, starting at time step 1
    # time step 0 is equal to the initial state of the obstacle
    dynamic_obstacle_trajectory = Trajectory(1, solution_trajectory)
    dynamic_obstacle_initial_state = solution_trajectory[0]

    # create the prediction using the trajectory and the shape of the obstacle
    dynamic_obstacle_shape = Rectangle(width=vehicle_params.w, length=vehicle_params.l)
    dynamic_obstacle_prediction = TrajectoryPrediction(
        dynamic_obstacle_trajectory, dynamic_obstacle_shape
    )

    # generate the dynamic obstacle according to the specification
    dynamic_obstacle_id = ego_id
    dynamic_obstacle_type = ObstacleType.CAR
    dynamic_obstacle = DynamicObstacle(
        obstacle_id=dynamic_obstacle_id,
        obstacle_type=dynamic_obstacle_type,
        obstacle_shape=dynamic_obstacle_shape,
        initial_state=dynamic_obstacle_initial_state,
        prediction=dynamic_obstacle_prediction,
    )

    # remove the obstacle since it was already added to the scenario
    scenario.remove_obstacle(obstacle=[scenario.obstacle_by_id(ego_id)])
    # add the dynamic obstacle to the scenario
    scenario.add_objects(dynamic_obstacle)

    # write new scenario
    # abuse the author tag to hold the ego vehicle's id
    fw = CommonRoadFileWriter(
        scenario=scenario, planning_problem_set=planning_problem_set, author=str(ego_id)
    )
    filename = scenario.benchmark_id + ".xml"
    fw.write_to_file(path + filename, OverwriteExistingFile.ALWAYS)


def is_in_interval(value: float, interval):
    """
    Check if the given value is in the given interval.

    Args:
        value (float): Given value.
        interval (Interval): Given interval.

    Returns:
        bool: True if the value is in the interval.
    """
    if interval.start < value < interval.end:
        return True
    else:
        return False


def get_unit_vector(angle: float):
    """
    Get the unit vector for a given angle.

    Args:
        angle (float): Considered angle.

    Returns:
        float: Unit vector of the considered angle
    """
    return np.array([np.cos(angle), np.sin(angle)])


def green_to_red_colormap():
    """Define a colormap that fades from green to red."""
    # This dictionary defines the colormap
    cdict = {
        "red": (
            (0.0, 0.0, 0.0),  # no red at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.8, 0.8),
        ),  # set to 0.8 so its not too bright at 1
        "green": (
            (0.0, 0.8, 0.8),  # set to 0.8 so its not too bright at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.0, 0.0),
        ),  # no green at 1
        "blue": (
            (0.0, 0.0, 0.0),  # no blue at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.0, 0.0),
        ),  # no blue at 1
    }

    # Create the colormap using the dictionary
    GnRd = colors.LinearSegmentedColormap("GnRd", cdict)

    return GnRd


def get_occluded_area(
    scenario: Scenario,
    time_step: float,
    ego_pos: List[float],
    sensor_radius: int = 50,
    wall_buffer: float = 0.0,
) -> Any:
    """
    Compute the area that is currently occluded by objects/vehicles or road geometry.

    Args:
        scenario: The considered scenario.
        time_step: The current time step of the scenario.
        ego_pos: The position of the ego vehicle.
        sensor_radius: The radius of the sensor in which things can be detected.
        wall_buffer: The wall buffer that defines if and how far the sensor can detect objects outside of the road.

    Returns: A MultiPolygon containing all occluded areas.

    """
    total_area = Point(ego_pos).buffer(sensor_radius)

    # Reduce visible area to lanelets
    for idx, lnlet in enumerate(scenario.lanelet_network.lanelets):
        pol_vertices = Polygon(
            np.concatenate((lnlet.right_vertices, lnlet.left_vertices[::-1]))
        )
        visible_lnlet = total_area.intersection(pol_vertices)

        if idx == 0:
            new_vis_area = visible_lnlet
        else:
            new_vis_area = new_vis_area.union(visible_lnlet)

    visible_area = new_vis_area

    # Enlarge visible area by wall buffer
    visible_area = visible_area.buffer(wall_buffer)

    # Substract areas that can not be seen due to geometry
    if visible_area.geom_type == 'MultiPolygon':
        allparts = [p.buffer(0) for p in visible_area.geometry]
        visible_area.geometry = shapely.ops.cascaded_union(allparts)

    points_vis_area = np.array(visible_area.exterior.xy).T

    for idx in range(points_vis_area.shape[0] - 1):
        vert_point1 = points_vis_area[idx]
        vert_point2 = points_vis_area[idx + 1]

        pol = _create_polygon_from_vertices(vert_point1, vert_point2, ego_pos)

        if pol.is_valid:
            visible_area = visible_area.difference(pol)

    occ_list = []
    # if occlusions through dynamic objects should be considered
    for dyn_obst in scenario.dynamic_obstacles:
        # check if obstacle is still there
        try:
            pos = dyn_obst.prediction.trajectory.state_list[time_step].position
            orientation = dyn_obst.prediction.trajectory.state_list[
                time_step
            ].orientation

        except IndexError:
            continue

        pos_point = Point(pos)
        # check if within sensor radius
        if pos_point.within(visible_area):
            # Substract occlusions from dynamic obstacles
            # Calculate corner points in world coordinates
            corner_points = _calc_corner_points(
                pos, orientation, dyn_obst.obstacle_shape
            )

            # Identify points for geometric projection
            r1, r2 = _identify_projection_points(corner_points, ego_pos)

            # Create polygon with points far away in the ray direction of ego pos
            r3 = r2 + __unit_vector(r2 - ego_pos) * sensor_radius
            r4 = r1 + __unit_vector(r1 - ego_pos) * sensor_radius

            occ_list.append(Polygon([r1, r2, r3, r4]))

    return unary_union(occ_list)


@enum.unique
class TUMColors(enum.Enum):
    """Enum containing TUM specific colors."""

    tum_blue = "#0065bd"
    tum_orange = "#e37222"
    tum_green = "#a2ad00"
    tum_gray = "#333333"
    tum_gray2 = "#7f7f7f"


# EOF
