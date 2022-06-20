
"""Class with relevant parameters harm estimation."""


class HarmParameters:
    """Harm parameters class."""

    def __init__(self):
        """
        Initialize the harm parameter class.

        Parameters:
            :param type (ObstacleType): type of object according to CommonRoad
                obstacle types
            :param protection (Boolean): displays if object has a protective
                crash structure
            :param mass (Float): mass of object in kg, estimated by type if
                exact value not existent
            :param velocity (Float): current velocity of object in m/s
            :param yaw (Float): current yaw of object in rad
            :param size (Float): size of object in square meters
                (length * width)
            :param harm (Float): estimated harm value
            :param prob (Float): estimated collision probability
            :param risk (Float): estimated collision risk (prob * harm)
        """
        # obstacle parameters
        self.type = None
        self.protection = None
        self.mass = None
        self.velocity = None
        self.yaw = None
        self.size = None
        self.harm = None
        self.prob = None
        self.risk = None
