"""Logging classes for Frenet planner."""
import json
import numpy as np
import time
import os
from pathlib import Path
from tqdm import tqdm  # noqa F401


class FrenetLogging:
    """Logging class that handles the setup and data-flow in order to write a log for the frenet planner in an iterative manner."""

    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(self, log_path: str) -> None:
        """Initialize Frenet Logger."""
        # Create directories
        Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
        # write header to logging file
        self.__log_path = log_path
        with open(self.__log_path, "w+") as fh:
            header = "simulation_step;time;valid_trajectories;invalid_trajectories;predictions;calc_time_avg;reach_set"

            fh.write(header)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def log_data(
        self,
        simulation_step: int,
        time: float,
        valid_trajectories: list,
        invalid_trajectories: list,
        predictions: dict,
        calc_time_avg: float,
        reach_set: dict
    ) -> None:
        """Write one line to the log file.

        :param time:             current time stamp (float time)
        """
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + json.dumps(simulation_step)
                + ";"
                + json.dumps(time)
                + ";"
                + json.dumps(valid_trajectories, default=default)
                + ";"
                + json.dumps(invalid_trajectories, default=default)
                + ";"
                + json.dumps(predictions, default=default)
                + ";"
                + json.dumps(calc_time_avg, default=default)
                + ";"
                + json.dumps(reach_set, default=default)
            )


class MessageLogging:
    """Logging class that handles the setup and data-flow in order to write a log for a trajectory planner in an iterative manner."""

    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(self, log_path: str) -> None:
        """Initilize message logging."""
        # write header to logging file
        self.__log_path = log_path
        with open(self.__log_path, "w+") as fh:
            header = "time;type;message"
            fh.write(header)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def log_message(self, time: float, msg_type: str, message: str) -> None:
        """Write one line to the log file.

        :param time:             current time stamp (float time)
        """
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time)
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )

    def warning(self, message: str) -> None:
        """Define message type warning.

        Args:
            message (str): [description]
        """
        msg_type = 'WARNING'
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time.time())
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )

    def info(self, message: str) -> None:
        """Define message type Info.

        Args:
            message (str): [description]
        """
        msg_type = 'INFO'
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time.time())
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )

    def debug(self, message: str) -> None:
        """Define messgae type debug.

        Args:
            message (str): [description]
        """
        msg_type = 'DEBUG'
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time.time())
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )

    def error(self, message: str) -> None:
        """Define message type error.

        Args:
            message (str): [description]
        """
        msg_type = 'error'
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time.time())
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )


def default(obj):
    """Handle numpy arrays when converting to json.

    Args:
        obj ([type]): [description]

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError('Not serializable (type: ' + str(type(obj)) + ')')


def get_data_from_line(file_path_in: str, line_num: int):
    """Read data from a single line of a file into the desired format.

    Args:
        file_path_in (str): [description]
        line_num (int): [description]

    Returns:
        [type]: [description]
    """
    line_num = max(1, line_num)
    # extract a certain line number (based on time_stamp)
    with open(file_path_in) as file:
        # get to top of file (1st line)
        file.seek(0)
        # get header (":-1" in order to remove tailing newline character)
        header = file.readline()[:-1]
        # extract line
        line = ""
        for _ in range(line_num):
            line = file.readline()

        # parse the data objects we want to retrieve from that line
        data = dict(zip(header.split(";"), line.split(";")))

        simulation_step = json.loads(data['simulation_step'])
        time = json.loads(data['time'])
        valid_trajectories = json.loads(data['valid_trajectories'])
        invalid_trajectories = json.loads(data['invalid_trajectories'])
        predictions = json.loads(data['predictions'])
        calc_time_avg = json.loads(data['calc_time_avg'])
        reach_set = json.loads(data['reach_set'])

        return (
            simulation_step,
            time,
            valid_trajectories,
            invalid_trajectories,
            predictions,
            calc_time_avg,
            reach_set
        )


def read_all_data(file_path_in, keys=None, zip_horz=False, verbose=False):
    """Read data from a file and return in it a dict.

    Args:
        file_path_in ([str]): [file path]
        keys ([type], optional): [description]. Defaults to None.
        zip_horz (bool, optional): [description]. Defaults to False.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if not verbose:
        tqdm = lambda x: x  # noqa F811
    with open(file_path_in) as f:
        total_lines = sum(1 for _ in f)

    total_lines = max(1, total_lines)

    all_data = None

    # extract a certain line number (based on time_stamp)

    with open(file_path_in) as file:
        # get to top of file (1st line)
        file.seek(0)
        # get header (":-1" in order to remove tailing newline character)
        header = file.readline()[:-1]
        # extract line
        line = ""
        for j in tqdm(range(total_lines - 1)):
            line = file.readline()

            if zip_horz:
                if all_data is None:
                    all_data = []
                    all_data = [header.split(";"), [None] * (total_lines - 1)]

                all_data[1][j] = tuple(json.loads(ll) for ll in line.split(";"))
            else:
                # parse the data objects we want to retrieve from that line
                data = dict(zip(header.split(";"), line.split(";")))
                if all_data is None:
                    if keys is None:
                        keys = data.keys()
                    all_data = {key: [0.0] * (total_lines - 1) for key in keys}
                for key in keys:
                    all_data[key][j] = json.loads(data[key])

    return all_data


def get_number_of_lines(file_path_in: str):
    """Get number of lines for a given file.

    Args:
        file_path_in (str): [file path]

    Returns:
        [int]: [number of lines]
    """
    with open(file_path_in) as file:
        row_count = sum(1 for row in file)

    return row_count


def log_param_dict(param_dict, logger):
    """Log parameter dict.

    Args:
        param_dict ([dict]): parameter dict
        logger ([FrenetLogging object]): object from FrenetLogging class
    """
    logger.info("=" * 40)
    for sec, dic in param_dict.items():
        logger.info(sec)
        for key, val in dic.items():
            logger.info(" - {}: {}".format(key, val))
    logger.info("=" * 40)
