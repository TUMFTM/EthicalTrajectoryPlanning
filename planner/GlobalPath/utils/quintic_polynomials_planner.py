"""

Quintic Polynomials Planner.

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Local Path planning And Motion Control For Agv In Positioning](http://ieeexplore.ieee.org/document/637936/)

"""

import math

import numpy as np

# parameter
MAX_T = 100.0  # maximum time to the goal [s]
MIN_T = 5.0  # minimum time to the goal[s]


class QuinticPolynomial:
    """Class to represent quintic polynomials."""

    def __init__(
        self,
        xs: float,
        vxs: float,
        axs: float,
        xe: float,
        vxe: float,
        axe: float,
        time: float,
    ):
        """
        Initialize a quintic polynomial.

        Args:
            xs (float): Position at the beginning of the polynomial.
            vxs (float): Velocity at the beginning of the polynomial.
            axs (float): Acceleration at the beginning of the polynomial.
            xe (float): Position at the end of the polynomial.
            vxe (float): Velocity at the end of the polynomial.
            axe (float): Acceleration at the end of the polynomial.
            time (float): Time at the end of the polynomial.
        """
        # calc coefficient of quintic polynomial
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array(
            [
                [time ** 3, time ** 4, time ** 5],
                [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                [6 * time, 12 * time ** 2, 20 * time ** 3],
            ]
        )

        b = np.array(
            [
                xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                vxe - self.a1 - 2 * self.a2 * time,
                axe - 2 * self.a2,
            ]
        )

        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        """
        Calculate position at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Position at time t.
        """
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t ** 2
            + self.a3 * t ** 3
            + self.a4 * t ** 4
            + self.a5 * t ** 5
        )

        return xt

    def calc_first_derivative(self, t):
        """
        Calculate first derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: First derivative at time t.
        """
        xt = (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t ** 2
            + 4 * self.a4 * t ** 3
            + 5 * self.a5 * t ** 4
        )

        return xt

    def calc_second_derivative(self, t):
        """
        Calculate second derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Second derivative at time t.
        """
        xt = (
            2 * self.a2
            + 6 * self.a3 * t
            + 12 * self.a4 * t ** 2
            + 20 * self.a5 * t ** 3
        )

        return xt

    def calc_third_derivative(self, t):
        """
        Calculate third derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Third derivative at time t.
        """
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


def quintic_polynomials_planner(
    start_point: float,
    syaw: float,
    goal_point: float,
    gyaw: float,
    sv: float = 1.0,
    sa: float = 0.0,
    gv: float = 1.0,
    ga: float = 0.0,
    max_accel: float = 1.0,
    max_jerk: float = 0.5,
    dt: float = 1.0,
):
    """
    Plan a quintic polynomial.

    input
        sx (float): start x position in m.
        sy (float): start y position in m.
        syaw (float): start yaw angle in rad.
        sa (float): start acceleration in m/s².
        gx (float): goal x position in m.
        gy (float): goal y position in m.
        gyaw (float): goal yaw angle in rad.
        ga (float): goal acceleration in m/s².
        max_accel (float): maximum acceleration in m/s².
        max_jerk (float): maximum jerk in m/s³.
        dt (float): time tick in s

    return
        float: Time result.
        float: x position result list.
        float: y position result list.
        float: Yaw angle result list.
        float: Velocity result list.
        float: Acceleration result list.

    """
    sx = start_point[0]
    sy = start_point[1]

    gx = goal_point[0]
    gy = goal_point[1]

    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    vxg = gv * math.cos(gyaw)
    vyg = gv * math.sin(gyaw)

    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)
    axg = ga * math.cos(gyaw)
    ayg = ga * math.sin(gyaw)

    time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

    for T in np.arange(MIN_T, MAX_T, MIN_T):
        xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

        time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

        for t in np.arange(0.0, T + dt, dt):
            time.append(t)
            rx.append(xqp.calc_point(t))
            ry.append(yqp.calc_point(t))

            vx = xqp.calc_first_derivative(t)
            vy = yqp.calc_first_derivative(t)
            v = np.hypot(vx, vy)
            yaw = math.atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)

            ax = xqp.calc_second_derivative(t)
            ay = yqp.calc_second_derivative(t)
            a = np.hypot(ax, ay)
            if len(rv) >= 2 and rv[-1] - rv[-2] < 0.0:
                a *= -1
            ra.append(a)

            jx = xqp.calc_third_derivative(t)
            jy = yqp.calc_third_derivative(t)
            j = np.hypot(jx, jy)
            if len(ra) >= 2 and ra[-1] - ra[-2] < 0.0:
                j *= -1
            rj.append(j)

        if (
            max([abs(i) for i in ra]) <= max_accel
            and max([abs(i) for i in rj]) <= max_jerk
        ):
            break

    # create the path
    for i in range(len(rx)):
        if i == 0:
            path = np.array([np.array([rx[i], ry[i]])])
        else:
            point = [rx[i], ry[i]]
            path = np.concatenate((path, np.array([point])))

    # return the path
    return path


# EOF
