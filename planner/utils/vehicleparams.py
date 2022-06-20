"""Module for easy creation of Vehicle Parameters."""


class VehicleParameters:
    """Vehicle parameters class."""

    def __init__(self, vehicle_string):
        """Initialize the vehicle parameter class."""
        # vehicle body dimensions
        self.l = None
        self.l_f = None
        self.l_r = None
        self.w = None

        # vehicle mass
        self.m = None
        # steering parameters
        self.steering = SteeringParameters()

        # longitudinal parameters
        self.longitudinal = LongitudinalParameters()

        if vehicle_string == "ford_escort":
            self.parameterize_ford_escort()
        elif vehicle_string == "bmw_320i":
            self.parameterize_bmw_320i()
        elif vehicle_string == "vw_vanagon":
            self.parameterize_vw_vanagon()
        else:
            raise ValueError("Value has to be ford_escort, bmw_320i or vw_vanagon")

    # TODO parameter in json files auslagern
    def parameterize_ford_escort(self):
        """Simplified parameter set of vehicle 1 (Ford Escort)."""
        # vehicle body dimensions
        self.l = 4.298  # vehicle length [m]
        self.l_f = (
            2.9 / 2.595
        )  # length from the center of gravity to the front axle [m]
        self.l_r = (
            4.95 / 2.595
        )  # length from the center of gravity to the rear axle [m]
        self.w = 1.674  # vehicle width [m]

        # vehicle mass
        self.m = 1050  # vehicle mass [kg]

        # steering constraints
        self.steering.min = -0.910  # minimum steering angle [rad]
        self.steering.max = 0.910  # maximum steering angle [rad]
        self.steering.v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering.v_max = 0.4  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.longitudinal.v_min = -13.9  # minimum velocity [m/s]
        self.longitudinal.v_max = 45.8  # maximum velocity [m/s]
        self.longitudinal.v_switch = 4.755  # switching velocity [m/s]
        self.longitudinal.a_max = 11.5  # maximum absolute acceleration [m/s^2]

        # lateral acceleration
        self.lateral_a_max = 10.0  # maximum lateral acceleartion [m/s^2]

    def parameterize_bmw_320i(self):
        """Simplified parameter set of vehicle 2 (BMW 320i)."""
        # vehicle body dimensions
        self.l = 4.508  # vehicle length [m] (with US bumpers)
        self.l_f = (
            3.793293 / 2.595
        )  # length from the center of gravity to the front axle [m]
        self.l_r = (
            4.667707 / 2.595
        )  # length from the center of gravity to the rear axle [m]
        self.w = 1.610  # vehicle width [m]

        # vehicle mass
        self.m = 1475  # vehicle mass [kg]

        # steering constraints
        self.steering.min = -1.066  # minimum steering angle [rad]
        self.steering.max = 1.066  # maximum steering angle [rad]
        self.steering.v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering.v_max = 0.4  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.longitudinal.v_min = -13.6  # minimum velocity [m/s]
        self.longitudinal.v_max = 50.8  # maximum velocity [m/s]
        self.longitudinal.v_switch = 7.319  # switching velocity [m/s]
        self.longitudinal.a_max = 11.5  # maximum absolute acceleration [m/s^2]

        # lateral acceleration
        self.lateral_a_max = 10.0  # maximum lateral acceleartion [m/s^2]

    def parameterize_vw_vanagon(self):
        """Simplified parameter set of vehicle 3 (VW Vanagon)."""
        # vehicle body dimensions
        self.l = 4.569  # vehicle length [m]
        self.l_f = (
            3.775563 / 2.595
        )  # length from the center of gravity to the front axle [m]
        self.l_r = (
            4.334437 / 2.595
        )  # length from the center of gravity to the rear axle [m]
        self.w = 1.844  # vehicle width [m]

        # vehicle mass
        self.m = 1450  # vehicle mass [kg]

        # steering constraints
        self.steering.min = -1.023  # minimum steering angle [rad]
        self.steering.max = 1.023  # maximum steering angle [rad]
        self.steering.v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering.v_max = 0.4  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.longitudinal.v_min = -11.2  # minimum velocity [m/s]
        self.longitudinal.v_max = 41.7  # maximum velocity [m/s]
        self.longitudinal.v_switch = 7.824  # switching velocity [m/s]
        self.longitudinal.a_max = 11.5  # maximum absolute acceleration [m/s^2]

        # lateral acceleration
        self.lateral_a_max = 10.0  # maximum lateral acceleartion [m/s^2]


class LongitudinalParameters:
    """Longitudinal parameters class."""

    def __init__(self):
        """Initialize the longitudinal parameter class."""
        # constraints regarding longitudinal dynamics
        self.v_min = None  # minimum velocity [m/s]
        self.v_max = None  # maximum velocity [m/s]
        self.v_switch = None  # switching velocity [m/s]
        self.a_max = None  # maximum absolute acceleration [m/s^2]


class SteeringParameters:
    """Steering parameters class."""

    def __init__(self):
        """Initialize the steering parameter class."""
        # constraints regarding steering
        self.min = None  # minimum steering angle [rad]
        self.max = None  # maximum steering angle [rad]
        self.v_min = None  # minimum steering velocity [rad/s]
        self.v_max = None  # maximum steering velocity [rad/s]
