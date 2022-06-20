"""
Implementation of the cost functions for occluded areas.

These functions are only called when the weight for 'visible_area' is >0
and a valid occlusion_mode is selected (currently 0, 1, or 2).
"""
from typing import List
import numpy as np
import shapely
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_helper_functions.sensor_model import (
    __unit_vector,
    _calc_corner_points,
    _create_polygon_from_vertices,
    _identify_projection_points,
)
from shapely.geometry import MultiPolygon, Point, Polygon

from EthicalTrajectoryPlanning.planner.Frenet.utils.helper_functions import distance


def calc_occluded_area_vs_velocity(
    traj: Trajectory,
    visible_area: Polygon,
    desired_visible_area: MultiPolygon,
    visibility_ratios: List,
    sensor_radius: float,
) -> float:
    """
    Calculate the costs for the occluded area with respect to the velocity of the given trajectory.

    Costs will be higher if the trajectory length is high and occluded area is high
    Args:
        traj (Trajectory): The considered frenet trajectory
        visible_area (Polygon): The occluded area.
        desired_visible_area (MultiPolygon): The desired visible area to feel safe.
        visibility_ratios ([float]): Ratios of visible vs occluded areas in dangerous zones.
        sensor_radius (float): The range of the sensor of the ego vehicle.

    Returns:
        float: Costs for the occluded area of the trajectory.
    """
    if (
        desired_visible_area is None
        or visible_area is None
        or visibility_ratios is None
    ):
        cost = 0
    else:
        # cost calculation based on ratios and traj_len (= velocity)
        # higher visibility_ratio = lower costs
        # higher velocity = higher costs
        weighted_vis_ratio = (
            visibility_ratios[0] * 1
            + visibility_ratios[1] * 1
            + visibility_ratios[-1] * 1
        ) / 3
        traj_len = distance(
            np.array([traj.x[0], traj.y[0]]), np.array([traj.x[-1], traj.y[-1]])
        )  # proxy for velocity
        cost = traj_len ** 2 / (
            weighted_vis_ratio * 75
        )  # some scaling for better balancing

    return cost


def calc_distance_to_occlusion(
    traj: Trajectory, visible_area: MultiPolygon, desired_visible_area: Polygon
) -> float:
    """
    Calculate the costs for occlusion based on the distance to the occluded areas.

    Args:
        traj (Trajectory): The considered frenet trajectory
        visible_area (MultiPolygon): The occluded area
        desired_visible_area (Polygon): The desired visible area to feel safe.
    Returns:
        float: Costs for the (minimum) distance of the trajectory to the occluded area
    """
    if desired_visible_area is None or visible_area is None:
        cost = 0
    else:
        occl_distance_list = []
        desired_distance_list = []
        for i in range(len(traj.x)):
            ego_pos = Point([traj.x[i], traj.y[i]])
            desired_distance_list.append(
                ego_pos.distance(desired_visible_area.boundary)
            )
            if visible_area.geom_type == 'GeometryCollection':
                for el in list(visible_area):
                    if el.geom_type == 'LineString':
                        occl_distance_list.append(ego_pos.distance(el))
                    else:
                        occl_distance_list.append(ego_pos.distance(el.boundary))
            else:
                occl_distance_list.append(ego_pos.distance(visible_area.boundary))

        min_dist_to_occl = min(occl_distance_list)
        min_dist_to_desired_vis = min(desired_distance_list)
        cost = (min_dist_to_desired_vis / (min_dist_to_occl + 0.01)) ** 0.5

    return cost


def get_visible_area(
    scenario: Scenario,
    time_step: float,
    ego_pos: List[float],
    sensor_radius: int = 50,
    wall_buffer: float = 0.0,
    desired_wall_buffer: float = 4.0,
) -> (MultiPolygon, Polygon):
    """
    Simulate a sensor model of a camera/lidar sensor and calculate desired/actual visible areas.

    Args:
        scenario (Scenario): The considered scenario.
        time_step (float): The current time step of the scenario.
        ego_pos (List[float]): The position of the ego vehicle.
        sensor_radius (int): The radius of the sensor in which things can be detected.
        wall_buffer (float): The wall buffer that defines how far the sensor can detect objects outside of the road.
        desired_wall_buffer (float): The desired wall buffer to feel save.

    Returns: Two Polygons
        visible_area (MultiPolygon):  the actual visible area based on road geometry, ego position and dynamic obstacles
        desired_visible_area (Polygon): the desired visible area
    """
    total_area = Point(ego_pos).buffer(sensor_radius)

    # Reduce visible area to lanelets
    for idx, lnlet in enumerate(scenario.lanelet_network.lanelets):
        pol_vertices = Polygon(
            np.concatenate((lnlet.right_vertices, lnlet.left_vertices[::-1]))
        )
        visible_lnlet = total_area.intersection(pol_vertices)

        if idx == 0:
            visible_lanelets = visible_lnlet
        else:
            visible_lanelets = visible_lanelets.union(visible_lnlet)

    # Enlarge visible area by wall buffer
    visible_area = visible_lanelets.buffer(wall_buffer)
    desired_visible_area = visible_lanelets.buffer(desired_wall_buffer)

    # Subtract areas that can not be seen due to geometry
    if visible_area.geom_type == 'MultiPolygon':
        allparts = [p.buffer(0) for p in visible_area.geometry]
        visible_area.geometry = shapely.ops.unary_union(allparts)

    points_vis_area = np.array(visible_area.exterior.xy).T

    for idx in range(points_vis_area.shape[0] - 1):
        vert_point1 = points_vis_area[idx]
        vert_point2 = points_vis_area[idx + 1]

        pol = _create_polygon_from_vertices(vert_point1, vert_point2, ego_pos)

        if pol.is_valid:
            visible_area = visible_area.difference(pol)

    # Subtract occluded areas from dynamic obstacles
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

            occlusion = Polygon([r1, r2, r3, r4])

            # Substract occlusion from visible area
            visible_area = visible_area.difference(occlusion)

    return visible_area, desired_visible_area
