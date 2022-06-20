#!/user/bin/env python

"""Calculate the collision probability of a trajectory and predictions."""

import os
import sys
import numpy as np
from scipy.stats import multivariate_normal
import commonroad_dc.pycrcc as pycrcc
from EthicalTrajectoryPlanning.risk_assessment.helpers.coll_prob_helpers import (
    distance,
    get_unit_vector,
)

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)


def get_collision_probability(traj, predictions: dict, vehicle_params, safety_margin=1.0):
    """
    Calculate the collision probabilities of a trajectory and predictions.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        predictions (dict): Predictions of the visible obstacles.
        vehicle_params (VehicleParameters): Parameters of the considered
            vehicle.

    Returns:
        dict: Collision probability of the trajectory per time step with the
            prediction for every visible obstacle.
    """
    obstacle_ids = list(predictions.keys())
    collision_prob_dict = {}
    safety_length = vehicle_params.l
    safety_width = vehicle_params.w

    for obstacle_id in obstacle_ids:
        mean_list = predictions[obstacle_id]['pos_list']
        cov_list = predictions[obstacle_id]['cov_list']
        yaw_list = predictions[obstacle_id]['orientation_list']
        length = predictions[obstacle_id]['shape']['length']
        probs = []
        for i in range(1, len(traj.x)):

            # only calculate probability as the predicted obstacle is visible
            if i < len(mean_list):

                # get the current position of the ego vehicle
                ego_pos = [traj.x[i], traj.y[i]]

                # get the mean and the covariance of the prediction
                mean = mean_list[i - 1]

                # get the position of the front and the back of the vehicle
                mean_front = mean + get_unit_vector(yaw_list[i]) * length / 2
                mean_back = mean - get_unit_vector(yaw_list[i]) * length / 2

                # if the distance between the vehicles is bigger than 5 meters,
                # the collision probability is zero
                # avoids huge computation times
                if (
                    min(
                        distance(ego_pos, mean),
                        distance(ego_pos, mean_front),
                        distance(ego_pos, mean_back),
                    )
                    > 5.0
                ):
                    prob = 0.0
                else:
                    cov = cov_list[i - 1]

                    # if the covariance is a zero matrix, the prediction is
                    # derived from the ground truth
                    # a zero matrix is not singular and therefore no valid
                    # covariance matrix
                    allcovs = [cov[0][0], cov[0][1], cov[1][0], cov[1][1]]
                    if all(covi == 0 for covi in allcovs):
                        cov = [[0.1, 0.0], [0.0, 0.1]]

                    prob = 0.0
                    means = [mean, mean_front, mean_back]

                    # the occupancy of the ego vehicle is approximated by three
                    # axis aligned rectangles
                    # get the center points of these three rectangles
                    center_points = get_center_points_for_shape_estimation(
                        length=safety_length,
                        width=safety_width,
                        orientation=traj.yaw[i],
                        pos=[traj.x[i], traj.y[i]],
                    )

                    # in order to get the cdf, the upper right point and the
                    # lower left point of every rectangle is needed
                    urs = []
                    lls = []
                    for center_point in center_points:
                        ur, ll = get_upper_right_and_lower_left_point(
                            center_point,
                            length=safety_length / 3,
                            width=safety_width,
                        )
                        urs.append(ur)
                        lls.append(ll)

                    # the probability distribution consists of the partial
                    # multivariate normal distributions
                    # this allows to consider the length of the predicted
                    # obstacle
                    # consider every distribution
                    for mu in means:
                        multi_norm = multivariate_normal(mean=mu, cov=cov)
                        # add the probability of every rectangle
                        for center_point_index in range(len(center_points)):
                            prob += get_prob_via_cdf(
                                multi_norm=multi_norm,
                                upper_right_point=urs[center_point_index],
                                lower_left_point=lls[center_point_index],
                            )

            else:
                prob = 0.0
            # divide by 3 since 3 probability distributions are added up and
            # normalize the probability
            probs.append(prob / 3)
        collision_prob_dict[obstacle_id] = probs

    return collision_prob_dict


def get_prob_via_cdf(
    multi_norm, upper_right_point: np.array, lower_left_point: np.array
):
    """
    Get CDF value.

    Get the CDF value for the rectangle defined by the upper right point and
    the lower left point.

    Args:
        multi_norm (multivariate_norm): Considered multivariate normal
            distribution.
        upper_right_point (np.array): Upper right point of the considered
            rectangle.
        lower_left_point (np.array): Lower left point of the considered
            rectangle.

    Returns:
        float: CDF value of the area defined by the upper right and the lower
            left point.
    """
    upp = upper_right_point
    low = lower_left_point
    # get the CDF for the four given areas
    cdf_upp = multi_norm.cdf(upp)
    cdf_low = multi_norm.cdf(low)
    cdf_comb_1 = multi_norm.cdf([low[0], upp[1]])
    cdf_comb_2 = multi_norm.cdf([upp[0], low[1]])
    # calculate the resulting CDF
    prob = cdf_upp - (cdf_comb_1 + cdf_comb_2 - cdf_low)

    return prob


def get_center_points_for_shape_estimation(
    length: float, width: float, orientation: float, pos: np.array
):
    """
    Get the 3 center points for axis aligned rectangles.

    Get the 3 center points for axis aligned rectangles that approximate an
    orientated rectangle.

    Args:
        length (float): Length of the oriented rectangle.
        width (float): Width of the oriented rectangle.
        orientation (float): Orientation of the oriented rectangle.
        pos (np.array): Center of the oriented rectangle.

    Returns:
        [np.array]: Array with 3 entries, every entry holds the center of one
            axis aligned rectangle.
    """
    # create the oriented rectangle
    obj = pycrcc.RectOBB(length / 2, width / 2, orientation, pos[0], pos[1])

    center_points = []
    obj_center = obj.center()
    # get the directional vector
    r_x = obj.r_x()
    # get the length
    a_x = obj.local_x_axis()
    # append three points (center point of the rectangle, center point of the
    # front third of the rectangle and center point of the back third of the
    # rectangle)
    center_points.append(obj_center)
    center_points.append(obj_center + r_x * (2 / 3) * a_x)
    center_points.append(obj_center - r_x * (2 / 3) * a_x)

    return center_points


def get_upper_right_and_lower_left_point(center: np.array, length: float, width: float):
    """
    Return upper right and lower left point of an axis aligned rectangle.

    Args:
        center (np.array): Center of the rectangle.
        length (float): Length of the rectangle.
        width (float): Width of the rectangle.

    Returns:
        np.array: Upper right point of the axis aligned rectangle.
        np.array: Lower left point of the axis aligned rectangle.
    """
    upper_right = [center[0] + length / 2, center[1] + width / 2]
    lower_left = [center[0] - length / 2, center[1] - width / 2]

    return upper_right, lower_left


def normalize_prob(prob: float):
    """
    Get a normalized value for the probability.

    Five partial linear equations are used to normalize the collision
    probability. This should avoid huge differences in the probabilities.
    Otherwise, low probabilities (e. g. 10⁻¹⁵⁰) would not be considered when
    other cost functions are used as well.
    This would result in a path planner, that does not consider risk at all if
    the risks appearing are pretty small.

    Args:
        prob (float): Initial probability.

    Returns:
        float: Resulting probability.
    """
    # dictionary with the factors of the linear equations
    factor_dict = {
        1: [0.6666666666666666, 0.33333333333333337],
        2: [1.1111111111111114, 0.28888888888888886],
        3: [10.101010101010099, 0.198989898989899],
        4: [1000.001000001, 0.0999998999999],
        5: [900000000.0000001, 0.01],
    }

    # normalize every probability with a suitable linear function
    if prob > 10 ** -1:
        return factor_dict[1][0] * prob + factor_dict[1][1]
    elif prob > 10 ** -2:
        return factor_dict[2][0] * prob + factor_dict[2][1]
    elif prob > 10 ** -4:
        return factor_dict[3][0] * prob + factor_dict[3][1]
    elif prob > 10 ** -10:
        return factor_dict[4][0] * prob + factor_dict[4][1]
    elif prob > 10 ** -70:
        return factor_dict[5][0] * prob + factor_dict[5][1]
    else:
        return 0.001


# EOF
