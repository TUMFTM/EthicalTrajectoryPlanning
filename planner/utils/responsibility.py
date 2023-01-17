"""Responsibility handling for ethical risk evaluation."""
import numpy as np
import pygeos


def calc_responsibility_reach_set(traj, ego_state, reach_set):
    """Calculate responsibilities using reachable sets.

    Args:
        traj (_type_): _description_
        ego_state (_type_): _description_
        reach_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    # prepare cache memory
    bool_contain_cache = []
    responsibility_cost = 0.0
    # time step
    dt = traj.t[1] - traj.t[0]
    # reach sets key list
    dict_items_list = list(reach_set.reach_sets[ego_state.time_step].items())

    for i in range(len(dict_items_list)):
        obj_id = dict_items_list[i][0]
        rs = dict_items_list[i][1]

        time_t_list = np.array([list(part_set.keys())[0] for part_set in rs])
        time_step_list = np.array(time_t_list / dt - 1, dtype=int)

        # ego pose
        point_array = np.stack((traj.x[time_step_list], traj.y[time_step_list]), axis=-1)
        ego_pos_array = pygeos.points(point_array)
        # ego_pos_array = pygeos.points(traj.x[time_step_list], traj.y[time_step_list])

        # prepare polygon array, make sure every polygon array has same length
        len_max = max(len(list(part_set.values())[0]) for part_set in rs)
        poly_array = [list(part_set.values())[0] for part_set in rs]
        poly_array_pad = polygon_padding(len_max, poly_array)

        # create polygon datatype
        obj_rs_array = pygeos.polygons(poly_array_pad)

        # whether ego point is contained in polygon
        bool_contain = np.array(pygeos.contains(obj_rs_array, ego_pos_array), dtype=int)
        # save cache
        bool_contain_cache.append(bool_contain)
        # ignore when time_t <= 0
        mask_array = np.array(time_t_list > 0, dtype=int)
        if 1 not in bool_contain * mask_array:
            responsibility_cost -= traj.obst_risk_dict[obj_id]

    return responsibility_cost, bool_contain_cache


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


def polygon_padding(max_poly_len, poly_array):
    """Polygon padding.

    Args:
        max_poly_len (_type_): _description_
        poly_array (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = np.zeros((len(poly_array), max_poly_len, 2))

    for i in range(len(poly_array)):
        if len(poly_array[i]) < max_poly_len:
            res[i][:len(poly_array[i])] = poly_array[i]
            # for j in range(len(poly_array[i]), max_poly_len):
            #     res[i][j] = poly_array[i][0]
            res[i][len(poly_array[i]):] = poly_array[i][-1]
        else:
            res[i] = poly_array[i]

    return res
