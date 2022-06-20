"""Responsibility handling for ethical risk evaluation."""
import numpy as np
from shapely.geometry import Point, Polygon


def calc_responsibility_reach_set(traj, ego_state, reach_set):
    """Calculate responsibilities using reachable sets.

    Args:
        traj (_type_): _description_
        ego_state (_type_): _description_
        reach_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    responsibility_cost = 0.0
    for obj_id, rs in reach_set.reach_sets[ego_state.time_step].items():
        # time_steps = [float(list(entry.keys())[0]) for entry in rs]
        responsibility = True
        for part_set in rs:
            time_t = list(part_set.keys())[0]
            if time_t <= 0:
                continue
            time_step = int(time_t / (traj.t[1] - traj.t[0]) - 1)

            ego_pos = Point(traj.x[time_step], traj.y[time_step])
            obj_rs = Polygon(list(part_set.values())[0])

            if obj_rs.contains(ego_pos):
                responsibility = False
                break

        if responsibility:
            responsibility_cost -= traj.obst_risk_dict[obj_id]

    return responsibility_cost


def assign_responsibility_by_action_space(scenario, ego_state, predictions):
    """Assign responsibility to prediction.

    Args:
        scenario ([type]): [description]
        ego_state ([type]): [description]
        predictions ([type]): [description]

    Returns:
        [type]: [description]
    """
    for pred_id in predictions:

        if check_if_inside180view(ego_state, predictions[pred_id]):
            predictions[pred_id]['responsibility'] = 0
        else:
            predictions[pred_id]['responsibility'] = 1

    return predictions


def check_if_inside180view(ego_state, prediction):
    """Check if predicted vehicle is within the 180 degree view of ego."""
    dx = prediction['pos_list'][0, 0] - ego_state.position[0]
    dy = prediction['pos_list'][0, 1] - ego_state.position[1]

    obst_ego_orientation = np.arctan2(dy, dx)

    if ego_state.orientation - (np.pi / 4) <= obst_ego_orientation <= ego_state.orientation + (np.pi / 4):
        return True

    else:
        return False
