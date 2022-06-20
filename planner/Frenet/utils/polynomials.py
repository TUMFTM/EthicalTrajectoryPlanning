"""This file provides the quintic and quartic polynomials for the frenet planner."""
# Standard imports


# Third party imports
import numpy as np


class quintic_polynomial:
    """Class to represent a quintic polynomial defined by start- position, velocity, acceleration and end- position, velocity, acceleration and end-time."""

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        """
        Initialize a quintic polynomial.

        Args:
            xs (float): Value at the start.
            vxs (float): Value of the first derivation at the start.
            axs (float): Value of the second derivation at the start.
            xe (float): Value at the end.
            vxe (float): Value of the first derivation at the end.
            axe (float): Value of the second derivation at the end.
            T (float): End time.
        """
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array(
            [
                [T ** 3, T ** 4, T ** 5],
                [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                [6 * T, 12 * T ** 2, 20 * T ** 3],
            ]
        )
        b = np.array(
            [
                xe - self.a0 - self.a1 * T - self.a2 * T ** 2,
                vxe - self.a1 - 2 * self.a2 * T,
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


class quartic_polynomial:
    """Class to represent a quartic polynomial defined by start- position, velocity, acceleration and end- velocity, acceleration and end-time."""

    def __init__(self, xs, vxs, axs, vxe, axe, T):
        """
        Initialize a quartic polynomial.

        Args:
            xs (float): Start value.
            vxs (float): Value of the first derivation at the start.
            axs (float): Value of the second derivation at the start.
            vxe (float): Value of the first derivation at the end.
            axe (float): Value of the second derivation at the end
            T (float): End time.
        """
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3], [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

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
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        """
        Calculate second derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Second derivative at time t.
        """
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        """
        Calculate third derivative at time t.

        Args:
            t (float): Time in s.

        Returns:
            float: Third derivative at time t.
        """
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt
