"""This module provides parent classes for easy evaluation of a planner."""
import sys
import pathlib
import random
import multiprocessing
import time
import json
import git

import progressbar
import numpy as np

if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


from commonroad_helper_functions.exceptions import (
    ExecutionTimeoutError,
    GoalReachedNotification,
)

from EthicalTrajectoryPlanning.planner.plannertools.scenario_handler import ScenarioHandler


class ScenarioEvaluator(ScenarioHandler):
    """Generic class for evaluating a scenario with a planner."""

    def eval_scenario(self, scenario_path):
        """WIP."""
        self.exec_timer.reset()
        self.scenario_path = self.path_to_scenarios.joinpath(scenario_path)
        start_time = time.time()
        with self.exec_timer.time_with_cm("total"):
            try:
                self._initialize()
                self._simulate()
            except GoalReachedNotification as excp:
                return_dict = {"success": True, "reason_for_failure": None}
                if "Goal reached but time exceeded!" in str(excp):
                    return_dict["reached_in_time"] = False
                elif "Goal reached in time!" in str(excp):
                    return_dict["reached_in_time"] = True
            except ExecutionTimeoutError as excp:
                return_dict = {"success": False, "reason_for_failure": str(excp)}
            except NotImplementedError as excp:
                raise excp
            except Exception as excp:
                # import traceback
                # traceback.print_exc()
                print(f"{scenario_path} >>> {str(excp)}")
                return_dict = {"success": False, "reason_for_failure": str(excp)}

                if "Simulation" in str(excp):
                    print(f"Stopping Evaluation, results not valid anymore due to simulation time out in {scenario_path}")
                    sys.exit()

            # TODO implement saving and animating scenario
            # self.postprocess()
        return_dict["scenario_path"] = scenario_path
        return_dict["exec_time"] = time.time() - start_time
        return_dict["harm"] = self.harm
        if not self.vel_list:
            return_dict["velocities"] = 0
        else:
            return_dict["velocities"] = np.mean(self.vel_list)
        return_dict["timesteps_agent"] = len(self.vel_list)
        if self.timing_enabled:
            return_dict["exec_times_dict"] = self.exec_timer.get_timing_dict()

        return return_dict


class DatasetEvaluator:
    """Class for creating a chunk of scenarios with a scenario evaluator."""

    def __init__(
        self, scenario_evaluator, eval_directory, limit_scenarios=None, disable_mp=False
    ):
        """Documenation in progress.

        Args:
            scenario_evaluator ([type]): [description]
            limit_scenarios ([int, list, none], optional):
                    int -> number of randomly picked scenarios is evaluated.
                    list[str] -> list of given scenario names is evaluated
                    None -> All scenarios are evaluated.
            disable_mp (bool, optional): [description]. Defaults to False.
        """
        self.scenario_evaluator = scenario_evaluator
        self.evaluation_function = self.scenario_evaluator.eval_scenario
        self.mp_disabled = disable_mp
        self.path_to_scenarios = self.scenario_evaluator.path_to_scenarios
        self.scenario_list = self._get_scenario_list(limit_scenarios)
        self.evaluation_time = None
        self.return_dicts = []
        self.eval_directory = eval_directory
        self.eval_files_generator = EvalFilesGenerator(self)

    def _get_scenario_list(self, limit_scenarios):
        """Create the dict with scenarios to evaluate.

        Returns:
            dict: dict with {"scenario_path": <number of timesteps in scenario>}
        """
        if isinstance(limit_scenarios, list):
            # Use debug scenarios
            scenario_index = limit_scenarios
            print("Using given scenarios!\n")
        else:
            # Get all scenario names from scenario repository
            print("Reading scenario names from repo:")
            scenario_index = [
                str(child.relative_to(self.path_to_scenarios))
                for child in self.path_to_scenarios.glob("**/*.xml")
            ]
            print(f"{len(scenario_index)} scenarios found!\n")

            # Remove Blacklist from scenario index
            scenario_index = [
                index
                for index in scenario_index
                if not any(
                    bl in index
                    for bl in self.scenario_evaluator.planner_creator.get_blacklist()
                )
            ]
            print(
                f"Remove blacklisted scenarios --> {len(scenario_index)} scenarios left\n"
            )

            # Check if the limit_scenarios key is a int
            if isinstance(limit_scenarios, int):
                scenario_index = random.sample(scenario_index, limit_scenarios)
                print(
                    f"Sampled {limit_scenarios} random scenarios from data set:\n{scenario_index}"
                )

        return scenario_index

    def eval_dataset(self):
        """WIP."""
        print("Evaluating Scenarios:")
        start_time = time.time()
        with progressbar.ProgressBar(max_value=len(self.scenario_list)).start() as pbar:
            if self.mp_disabled:
                # Calculate single threaded
                self._loop_with_single_processing(pbar)
            else:
                # use multiprocessing
                self._loop_with_with_mulitprocessing(pbar)
        self.evaluation_time = time.time() - start_time
        self.eval_files_generator.create_eval_files()

    def _loop_with_single_processing(self, pbar):
        for scenario_path in self.scenario_list:
            return_dict = self.evaluation_function(scenario_path)
            self._process_return_dict(return_dict)
            pbar.update(pbar.value + 1)

    def _loop_with_with_mulitprocessing(self, pbar):
        cpu_count = 10  # multiprocessing.cpu_count()
        # create worker pool
        with multiprocessing.Pool(processes=cpu_count) as pool:
            # Use imap unordered to parelly eval the results
            for return_dict in pool.imap_unordered(
                self.evaluation_function, self.scenario_list
            ):
                self._process_return_dict(return_dict)
                pbar.update(pbar.value + 1)

    def _process_return_dict(self, return_dict):
        """WIP."""
        self.return_dicts.append(return_dict)


class EvalFilesGenerator:
    """Helper class for a dataset evaluator to generate eval files."""

    def __init__(self, dataset_evaluator):
        """Documentation in progress."""
        self.dataset_evaluator = dataset_evaluator

    @property
    def _number_processed(self):
        return len(self.dataset_evaluator.return_dicts)

    @property
    def _number_successful(self):
        num = 0
        for return_dict in self.dataset_evaluator.return_dicts:
            if return_dict["success"]:
                num += 1
        return num

    @property
    def _number_successful_in_time(self):
        num = 0
        for return_dict in self.dataset_evaluator.return_dicts:
            if return_dict["success"]:
                if return_dict["reached_in_time"]:
                    num += 1
        return num

    @property
    def _completion_rate(self):
        return 100 * self._number_successful / self._number_processed

    @property
    def _in_time_completion_rate(self):
        return 100 * self._number_successful_in_time / self._number_processed

    @property
    def _avg_exection_time_per_scenario(self):
        return np.mean(
            [
                return_dict["exec_time"]
                for return_dict in self.dataset_evaluator.return_dicts
            ]
        )

    @property
    def _avg_exection_time_per_successfull_scenario(self):
        if self._number_successful > 0:
            return np.mean(
                [
                    return_dict["exec_time"]
                    for return_dict in self.dataset_evaluator.return_dicts
                    if return_dict["success"] is True
                ]
            )
        else:
            return 0.0

    @property
    def _fail_log(self):
        fail_list = []
        for return_dict in self.dataset_evaluator.return_dicts:
            if return_dict["success"] is False:
                fail_reason = return_dict["reason_for_failure"]
                # Truncate error messages longer than 100 chars
                if len(fail_reason) > 100:
                    fail_reason = fail_reason[:100] + " ..."
                fail_list.append(return_dict["scenario_path"] + " >>> " + fail_reason)
        return fail_list

    # not used at the moment
    @property
    def _avg_velocity_driven(self):
        v = []
        for i in range(len(self.dataset_evaluator.return_dicts)):
            if self.dataset_evaluator.return_dicts[i]['success'] is True:
                v.append(self.dataset_evaluator.return_dicts[0]['velocities'])

        return np.mean(v)

    @property
    def _avg_velocity(self):
        v = []
        for i in range(len(self.dataset_evaluator.return_dicts)):
            if self.dataset_evaluator.return_dicts[i]['success'] is True:
                v.append(self.dataset_evaluator.return_dicts[0]['velocities'])
            else:
                v.append(0)

        return np.mean(v)

    @property
    def _avg_velocity_weighted(self):
        v = []
        for i in range(len(self.dataset_evaluator.return_dicts)):
            if self.dataset_evaluator.return_dicts[i]['success'] is True:
                v.append(self.dataset_evaluator.return_dicts[0]['velocities'])
        w = [
            return_dict["timesteps_agent"]
            for return_dict in self.dataset_evaluator.return_dicts
        ]
        s = np.sum(
            [
                return_dict["timesteps_agent"]
                for return_dict in self.dataset_evaluator.return_dicts
            ]
        )
        velocity = 0
        for x in range(len(v)):
            velocity += (v[x] * w[x]) / s
        return velocity

    def create_eval_files(self):
        """WIP."""
        self._create_eval_statistic()
        self._create_completion_list()
        self._create_harm_evaluation()
        self._create_eval_statistic_exec_times_dict()
        self._store_weights()

    def _create_completion_list(self):
        """WIP."""
        file_path = self.dataset_evaluator.eval_directory.joinpath(
            "scenario_completion_list"
        ).with_suffix(".json")

        eval_dict = {"completed": [], "failed": [], "invalid": []}
        for return_dict in self.dataset_evaluator.return_dicts:
            if return_dict["success"]:
                eval_dict["completed"].append(return_dict["scenario_path"])
            elif "successor" in return_dict["reason_for_failure"]:
                eval_dict["invalid"].append(return_dict["scenario_path"])
            else:
                eval_dict["failed"].append(return_dict["scenario_path"])

        with open(file_path, "w") as write_file:
            json.dump(eval_dict, write_file, indent=4)

    def _create_eval_statistic(self):
        file_path = self.dataset_evaluator.eval_directory.joinpath(
            "planner_statistic"
        ).with_suffix(".log")
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        eval_list = [f"Commit: {sha}"]
        eval_list.append(f"Number of Evaluated Scenarios: {self._number_processed}")
        eval_list.append("")
        eval_list.append(
            f"Number of successfully driven scenarios: {self._number_successful}"
        )
        eval_list.append(f"Completion rate: {self._completion_rate:.2f} %")
        eval_list.append("")
        eval_list.append(
            f"Number of scenarios completed in time: {self._number_successful_in_time}"
        )
        eval_list.append(
            (f"In time completion rate: {self._in_time_completion_rate:.2f} %")
        )
        eval_list.append(f"Average velocity: {round(self._avg_velocity,2)} m/s")
        eval_list.append(
            f"Weighted average velocity: {round(self._avg_velocity_weighted,2)} m/s"
        )
        eval_list.append("\n")
        hours = int(self.dataset_evaluator.evaluation_time // 3600)
        mins = int((self.dataset_evaluator.evaluation_time % 3600) // 60)
        secs = int((self.dataset_evaluator.evaluation_time % 3600) % 60)
        eval_list.append(f"Execution time: {hours} h  {mins} min  {secs} sec")
        eval_list.append(
            f"Average execution time per scenario: {self._avg_exection_time_per_scenario:.2f} sec"
        )
        eval_list.append(
            "Average execution time per successfully completed scenario: "
            + f"{self._avg_exection_time_per_successfull_scenario:.2f} sec"
        )
        eval_list.append("\n")
        eval_list.append("Failure log:")
        eval_list.append("-------------------------------------------")
        eval_list.extend(self._fail_log)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as file_obj:
            file_obj.write("\n".join(eval_list))

    def _create_eval_statistic_exec_times_dict(self):
        # Check if there are exec_times_dicts
        if "exec_times_dict" in self.dataset_evaluator.return_dicts[0]:
            # Extract the dicts
            file_path = self.dataset_evaluator.eval_directory.joinpath(
                "exec_timing"
            ).with_suffix(".json")

            exec_times_dicts = [
                return_dict["exec_times_dict"]
                for return_dict in self.dataset_evaluator.return_dicts
            ]
            # Merge list of dicts to dict of list
            dict_merged = {}
            for key in exec_times_dicts[0]:
                for exec_times_dict in exec_times_dicts:
                    if key in exec_times_dict:
                        if key not in dict_merged:
                            dict_merged[key] = []
                        dict_merged[key].append(exec_times_dict[key])

            # Merge the list of lists together
            dict_merged = {
                key: [
                    inner_item for outer_item in list_item for inner_item in outer_item
                ]
                for key, list_item in dict_merged.items()
            }

            def eval_merged_dict(dict_merged):
                out_dict = {}
                total_time = sum(dict_merged["total"])
                for key, item in dict_merged.items():
                    eval_list = []
                    eval_list.append(
                        f"Percentage from total: {100*sum(item)/total_time:.3f} %"
                    )
                    eval_list.append(f"Total time: {sum(item):.4f} s")
                    eval_list.append(f"Number of calls: {len(item)}")
                    eval_list.append(
                        f"Avg exec time per call: {sum(item)/len(item):.6f}"
                    )
                    out_dict[key] = " || ".join(eval_list)
                return out_dict

            evaluated_merged_dict = eval_merged_dict(dict_merged)

            def group_dict_recursive(input_rec_dict):

                error_msg = (
                    "Do not use a label that has a parent used for timing\n\n"
                    + "Minimal example to reproduce this error:\n"
                    + """>>> with timer.time_with_cm("super/stupid"):\n"""
                    + ">>>     pass\n"
                    + """>>> with timer.time_with_cm("super/stupid/example"):\n"""
                    + ">>>     pass\n"
                )
                working_dict = {}
                for key, item in input_rec_dict.items():
                    split_key = key.split("/", 1)
                    if len(split_key) == 1:
                        if split_key[0] in working_dict:
                            raise Exception(
                                error_msg + f"\nKey that threw error: {key}"
                            )
                        else:
                            working_dict[split_key[0]] = item
                    else:
                        if split_key[0] in working_dict:
                            try:
                                working_dict[split_key[0]][split_key[1]] = item
                            except TypeError as excp:
                                raise Exception(
                                    error_msg + f"\nKey that threw error: {key}"
                                ) from excp
                        else:
                            working_dict[split_key[0]] = {split_key[1]: item}

                for key, item in working_dict.items():
                    if isinstance(item, dict):
                        working_dict[key] = group_dict_recursive(item)

                return working_dict

            grouped_dict = group_dict_recursive(evaluated_merged_dict)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as file_obj:
                json.dump(grouped_dict, file_obj, indent=6)

    def _create_harm_evaluation(self):
        harm = {
            "Ego": 0.0,
            "Unknown": 0.0,
            "Car": 0.0,
            "Truck": 0.0,
            "Bus": 0.0,
            "Bicycle": 0.0,
            "Pedestrian": 0.0,
            "Priority_vehicle": 0.0,
            "Parked_vehicle": 0.0,
            "Construction_zone": 0.0,
            "Train": 0.0,
            "Road_boundary": 0.0,
            "Motorcycle": 0.0,
            "Taxi": 0.0,
            "Building": 0.0,
            "Pillar": 0.0,
            "Median_strip": 0.0,
            "Total": 0.0,
        }
        # Check if collision is available
        if "harm" in self.dataset_evaluator.return_dicts[0]:
            for return_dict in self.dataset_evaluator.return_dicts:
                for key in return_dict["harm"].keys():
                    harm[key] += return_dict["harm"][key]

        # Write harm dict to json
        file_path = self.dataset_evaluator.eval_directory.joinpath("harm").with_suffix(
            ".json"
        )

        with open(file_path, "w") as output:
            json.dump(harm, output, indent=6)

    def _store_weights(self):
        file_path = self.dataset_evaluator.eval_directory.joinpath("weights").with_suffix(
            ".json"
        )
        weights = self.dataset_evaluator.scenario_evaluator.planner_creator.weights

        with open(file_path, "w") as output:
            json.dump(weights, output, indent=6)
