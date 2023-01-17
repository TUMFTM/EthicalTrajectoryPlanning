"""Analyze and visualize logfiles."""

import os
import sys
import pprint
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import argparse
from scipy import stats
from statsmodels.formula.api import ols

mopl_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
sys.path.append(mopl_path)

from EthicalTrajectoryPlanning.planner.Frenet.utils.logging import read_all_data
from EthicalTrajectoryPlanning.planner.Frenet.utils.visualization_utils import SliderGroup
from EthicalTrajectoryPlanning.planner.Frenet.utils.visualization import (
    draw_frenet_trajectories,
    draw_reach_sets,
    draw_scenario,
)
from EthicalTrajectoryPlanning.planner.planning import add_ego_vehicles_to_scenario
from EthicalTrajectoryPlanning.planner.utils.vehicleparams import VehicleParameters
from EthicalTrajectoryPlanning.risk_assessment.helpers.coll_prob_helpers import distance


class FrenetLogVisualizer:
    """Class to handle visualization functions for frenet logs."""

    def __init__(self, logfile, visualize=True, verbose=True):
        """Initialize visualizer.

        Args:
            logfile ([type]): [description]
            visualize (bool, optional): [description]. Defaults to True.
            verbose (bool, optional): [description]. Defaults to True.
        """
        if logfile is None:
            logfile = self._ask_for_logfile()

        self._load_log_data(logfile)
        self.scenario_id = logfile.split("/")[-1].split(".")[0]
        self.scenario_path = self._search_for_scenario_path(self.scenario_id)

        self.vehicle_params = VehicleParameters("bmw_320i")  # TODO

        self._load_scenario(self.scenario_id)
        self._add_ego_to_scenario()

        self.risk_keys = ["bayes", "equality", "maximin", "ego", "responsibility"]
        self.weights = self.best_traj_list[0]["cost_dict"]["weights"]
        self.plot_frenet = True
        self.plot_reach = False

        if visualize:
            self._init_figure()

    def _ask_for_logfile(self):

        print(
            "No logfile provided. Use --logfile <path_to_logfile> to provide a specific logfile or choose from below."
        )

        logdir = "./planner/Frenet/results/logs"
        paths = sorted(Path(logdir).iterdir(), key=os.path.getmtime, reverse=True)

        print(
            "Displaying the most recent logfiles. Choose by entering the number or hit enter for the most recent log:"
        )
        for i, p in enumerate(paths[:9]):
            print(f"[{i + 1}] for {p}")

        chosen_number = input()
        if chosen_number == "":
            return paths[0]._str
        else:
            return paths[int(chosen_number) - 1]._str

    def _search_for_scenario_path(self, scenario_id):
        scenario_dir = os.path.join(
            mopl_path,
            "commonroad-scenarios",
        )

        filelist = []
        # Get all files in the directory and the subdirectories
        for root, dirs, files in os.walk(scenario_dir):
            for file in files:
                filelist.append(os.path.join(root, file))

        for filepath in filelist:
            if scenario_id in filepath:
                return filepath
        else:
            raise FileNotFoundError

    def _load_log_data(self, logfile):
        """Load the data from the logfiles.

        If logdir is specified, the
        logs from that folder are loaded. If it is not specified, the
        very last available log file is chosen.

        args:
            logdir: (str), the dir that contains the logfiles to load.
        """
        # searching for the files to open:
        if os.path.exists(logfile):

            # Load all the data into dict
            _, self._all_log_data = read_all_data(logfile, zip_horz=True)

            # Preprocess for best_traj
            self.best_traj_list = []

            for data in self._all_log_data:
                (
                    time_step,
                    time,
                    ft_list_valid,
                    ft_list_invalid,
                    predictions,
                    calc_time_avg,
                    reach_set,
                ) = data

                self.best_traj_list.append((ft_list_valid + ft_list_invalid)[0])

            # Get number of lines
            self._no_lines_data = len(self._all_log_data)

        else:
            print(f"Logfile {logfile} does not exist.")
            sys.exit()

    def _load_scenario(self, scneario_id):

        self.scenario, self.planning_problem_set = CommonRoadFileReader(
            self.scenario_path
        ).open()

    def _add_ego_to_scenario(self):
        (
            self.scenario,
            self.agent_planning_problem_id_assignment,
        ) = add_ego_vehicles_to_scenario(
            scenario=self.scenario,
            planning_problem_set=self.planning_problem_set,
            vehicle_params=self.vehicle_params,
        )

    def _draw_scenario(
        self,
        t,
        trajectories,
        predictions,
        reach_set,
        plot_trajectories=False,
        plot_reach_set=False,
    ):
        """
        Draw the scenario.

        args:
            t: (int), time step.
            trajectories: frenet trajectories.
            predictions: predictions.
            reach_set: reachable sets.
            plot_trajectories: (bool) true to plot trajectories.
            plot_reach_set: (bool) true to plot reachable sets.
        """
        planning_problem = list(
            self.planning_problem_set.planning_problem_dict.values()
        )[0]

        trajectories = [AttrDict(traj) for traj in trajectories]

        draw_scenario(
            self.scenario,
            t,
            planning_problem=planning_problem,
            traj=trajectories[0],
            ax=self._ax1,
            picker=True,
            show_label=True,
        )

        if plot_reach_set:
            draw_reach_sets(
                traj=trajectories[0],
                reach_set=reach_set,
                ax=self._ax1,
            )

        if plot_trajectories:
            draw_frenet_trajectories(
                self.scenario,
                t,
                planning_problem=planning_problem,
                all_traj=trajectories,
                traj=trajectories[0],
                predictions=predictions,
                ax=self._ax1,
                picker=True,
                show_label=True,
                live=False,
            )

        plt.pause(0.0001)

    def _init_figure(self):
        """Initialize the figure."""
        self._fig, self._axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 9))

        gs = self._axs[0, 0].get_gridspec()
        # remove the underlying axes
        for ax in self._axs[:, 0]:
            ax.remove()
        self._ax1 = self._fig.add_subplot(gs[:, 0])
        self._ax2 = self._axs[0, 1]
        self._ax3 = self._axs[1, 1]

        # setting axis labels of ax1:
        self._ax1.set_xlabel("X [m]")
        self._ax1.set_ylabel("Y [m]")

        self._ax1.axis('equal')

        plt.subplots_adjust(bottom=0.25)

        self.time_list = [i * self.scenario.dt for i in range(len(self._all_log_data))]

        self._setup_risk_plot()
        self._setup_risk_cost_plot()

        # slider for the global simulation time:
        self._global_slider = SliderGroup(
            fig=self._fig,
            left=0.25,
            bottom=0.1,
            width=0.6,
            height=0.04,
            max_val=self._no_lines_data - 1,
            step=1,
            text="Global Timestep",
            callback=self._update,
        )

        # Check button

        rax = plt.axes([0.02, 0.1, 0.1, 0.04])
        check = CheckButtons(rax, ["Show IDs"], [True])

        check.on_clicked(self._show_unshow_ids)

        # slider for the local simulation time within a prediction:
        # self._local_slider = SliderGroup(
        #     fig=self._fig,
        #     left=0.2,
        #     bottom=0.05,
        #     width=0.6,
        #     height=0.04,
        #     max_val=10,
        #     step=1,
        #     text="Local Timestep",
        #     callback=self._update_local_timestep_markers,
        # )

        # connecting the pick event to the _on_pick() method:
        self._fig.canvas.mpl_connect('pick_event', self._on_pick)

        # Checkbutton for selection
        rax = self._fig.add_axes([0.01, 0.89, 0.12, 0.1], facecolor='lightgrey')
        self.__check_selection = CheckButtons(
            rax, ('frenet trajectories', 'reach set'), actives=[True, False]
        )
        self.__check_selection.on_clicked(self._visualize_selected)

    def _setup_risk_plot(self):

        # Setup Risk plot
        ego_risk_list = []
        obst_risk_dict = {}

        all_obst_ids = []
        for best_traj in self.best_traj_list:
            for obst_id in best_traj["obst_risk_dict"]:
                if int(obst_id) not in all_obst_ids:
                    all_obst_ids.append(int(obst_id))

        for obst_id in all_obst_ids:
            obst_risk_dict[obst_id] = []

        for best_traj in self.best_traj_list:
            ego_risk_list.append(sum(best_traj["ego_risk_dict"].values()))
            for obst_id in all_obst_ids:
                if str(obst_id) not in best_traj["obst_risk_dict"]:
                    obst_risk_dict[obst_id].append(0)
                else:
                    obst_risk_dict[obst_id].append(
                        best_traj["obst_risk_dict"][str(obst_id)]
                    )

        self._ax2.plot(self.time_list, ego_risk_list, label="Ego Risk")
        for key in obst_risk_dict:
            self._ax2.plot(self.time_list, obst_risk_dict[key], label=f"Obstacle {key}")

        self._ax2.set_ylabel("Risks")
        self._ax2.legend()

        # Cursor
        y_min, y_max = self._ax2.get_ylim()
        self._cursor_ax2 = self._ax2.plot([0, 0], [y_min, y_max], color="red")

    def _setup_risk_cost_plot(self):

        # Get weights from anywhere
        weights = self.weights
        weights["responsibility"] *= weights["bayes"]

        # Create risk cost dict
        self.weighted_cost_dict = {}
        for key in weights:
            if weights[key] != 0 and "risk" not in key:
                self.weighted_cost_dict[key] = []

        for key in self.weighted_cost_dict:
            for best_traj in self.best_traj_list:
                if key in self.risk_keys:
                    self.weighted_cost_dict[key].append(
                        best_traj["cost_dict"]["risk_cost_dict"][key]
                        * weights[key]
                        * weights["risk_cost"]
                    )
                else:
                    self.weighted_cost_dict[key].append(
                        best_traj["cost_dict"]["unweighted_cost"][key] * weights[key]
                    )

            self._ax3.plot(
                self.time_list,
                self.weighted_cost_dict[key],
                label=f"{key} (weight = {weights[key]})",
            )

        self._ax3.legend()
        self._ax3.set_xlabel("Time in s")
        self._ax3.set_ylabel("Risk Costs")

        # Cursor
        y_min, y_max = self._ax3.get_ylim()
        self._cursor_ax3 = self._ax3.plot([0, 0], [y_min, y_max], color="red")

    def _on_pick(self, event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()

            selected_traj = self._find_trajectory(xdata, ydata)
            pprint.pprint(selected_traj["cost_dict"])
        else:
            print("no line")

    def _show_unshow_ids(self, label):
        print("Hee we are")
        plt.draw()

    def _find_trajectory(self, xdata, ydata):

        for traj in self.ft_list_valid + self.ft_list_invalid:
            if traj["x"] == list(xdata) and traj["y"] == list(ydata):
                return traj

    def _update(self, val):
        """Call to update the plot i.e when the slider has changed."""
        (
            time_step,
            time,
            self.ft_list_valid,
            self.ft_list_invalid,
            predictions,
            calc_time_avg,
            reach_set,
        ) = self._all_log_data[int(val)]

        self._draw_scenario(
            time_step,
            self.ft_list_valid + self.ft_list_invalid,
            predictions,
            reach_set,
            plot_trajectories=self.plot_frenet,
            plot_reach_set=self.plot_reach,
        )

        self._cursor_ax2[0].set_xdata([val * self.scenario.dt] * 2)
        self._cursor_ax3[0].set_xdata([val * self.scenario.dt] * 2)

        self._fig.canvas.draw()

    def visualize(self):
        """Visualize func."""
        self._update(0)
        plt.show()

    def correlation_matrix(self, plot=True):
        """Calculate and visualize correlation matrix."""
        all_weighted_cost_dict = self._get_all_weighted_cost_dict()

        corr_mat = np.zeros(
            (len(all_weighted_cost_dict), len(all_weighted_cost_dict))
        )
        for key1, x in enumerate(all_weighted_cost_dict.values()):
            for key2, y in enumerate(all_weighted_cost_dict.values()):
                correlation, p_value = stats.pearsonr(x, y)
                corr_mat[key1, key2] = correlation

        if plot:
            self._draw_corr_mat(corr_mat, all_weighted_cost_dict)

        return corr_mat, list(all_weighted_cost_dict.keys())

    def multi_correlation(self, use_keys=None):
        """Calculate coefficients of multiple correlation.

        See: https://en.wikipedia.org/wiki/Coefficient_of_multiple_correlatio
        """
        all_weighted_cost_dict = self._get_all_weighted_cost_dict()

        if use_keys is not None:
            all_weighted_cost_dict = {k: v for k, v in all_weighted_cost_dict.items() if k in use_keys}

        multi_corr_dict = analyze_multi_correlation(all_weighted_cost_dict)

        return multi_corr_dict

    def distance_matrix(self, plot=False):
        """Calculate distances for the best chosen trajectory according to each principle."""
        self.long_dict = {}
        self.lat_dict = {}

        weights = self.weights
        weights["total"] = -1

        for key in weights:
            if weights[key] != 0 and "risk" not in key:
                self.long_dict[key] = []
                self.lat_dict[key] = []

                for data in self._all_log_data:
                    (_, _, ft_list_valid, _, _, _, _) = data

                    best_traj = self._get_best_traj_for_principle(ft_list_valid, key)
                    self.long_dict[key].append(best_traj["s"][-1])
                    self.lat_dict[key].append(best_traj["d"][-1])

        long_mat = np.zeros(
            (len(self.long_dict), len(self.long_dict))
        )
        long_std = np.zeros(
            (len(self.long_dict), len(self.long_dict))
        )
        lat_mat = np.zeros(
            (len(self.lat_dict), len(self.lat_dict))
        )
        lat_std = np.zeros(
            (len(self.lat_dict), len(self.lat_dict))
        )
        dist_mat = np.zeros(
            (len(self.lat_dict), len(self.lat_dict))
        )
        dist_std = np.zeros(
            (len(self.lat_dict), len(self.lat_dict))
        )

        for key1, x in enumerate(self.long_dict.values()):
            for key2, y in enumerate(self.long_dict.values()):
                diff = [a - b for a, b in zip(x, y)]
                long_mat[key1, key2] = np.mean(np.abs(diff))
                long_std[key1, key2] = np.std(np.abs(diff))

        for key1, x in enumerate(self.lat_dict.values()):
            for key2, y in enumerate(self.lat_dict.values()):
                diff = [a - b for a, b in zip(x, y)]
                lat_mat[key1, key2] = np.mean(np.abs(diff))
                lat_std[key1, key2] = np.std(np.abs(diff))

        for key1, (x, y) in enumerate(zip(self.lat_dict.values(), self.long_dict.values())):
            for key2, (xx, yy) in enumerate(zip(self.lat_dict.values(), self.long_dict.values())):
                diff = [distance([a, b], [aa, bb]) for a, b, aa, bb in zip(x, y, xx, yy)]
                dist_mat[key1, key2] = np.mean(np.abs(diff))
                dist_std[key1, key2] = np.std(np.abs(diff))

        if plot:
            self._draw_corr_mat(lat_mat, self.lat_dict, stdw_mat=lat_std, inverse=True, title="Lateral Distance Matrix")
            self._draw_corr_mat(long_mat, self.long_dict, stdw_mat=long_std, inverse=True, title="Longitudinal Distance Matrix")
            self._draw_corr_mat(dist_mat, self.lat_dict, stdw_mat=dist_std, inverse=True, title="Total Distance Matrix")
            plt.show()

        return long_mat, lat_mat, dist_mat

    def _get_all_weighted_cost_dict(self):
        all_weighted_cost_dict = {}
        weights = self.weights

        for key in weights:
            if weights[key] > 0 and "risk" not in key:
                all_weighted_cost_dict[key] = []

        for key in all_weighted_cost_dict:
            for data in self._all_log_data:
                (_, _, ft_list_valid, _, _, _, _) = data

                for traj in ft_list_valid:
                    if key in self.risk_keys:
                        all_weighted_cost_dict[key].append(
                            traj["cost_dict"]["risk_cost_dict"][key]
                            * weights[key]
                            * weights["risk_cost"]
                        )
                    else:
                        all_weighted_cost_dict[key].append(
                            traj["cost_dict"]["unweighted_cost"][key] * weights[key]
                        )

        return all_weighted_cost_dict

    def _get_best_traj_for_principle(self, traj_list, cost_key):

        curr_best_weighted_cost = np.inf
        weights = self.weights

        for traj in traj_list:
            if cost_key in self.risk_keys:
                weighted_cost = traj["cost_dict"]["risk_cost_dict"][cost_key] * weights[cost_key] * weights["risk_cost"]

            elif cost_key == "total":
                weighted_cost = traj["cost"]
            else:
                weighted_cost = traj["cost_dict"]["unweighted_cost"][cost_key] * weights[cost_key]

            if weighted_cost < curr_best_weighted_cost:
                curr_best_weighted_cost = weighted_cost
                curr_best_traj = traj

        return curr_best_traj

    def _draw_corr_mat(self, corr_mat, raw_dict, stdw_mat=None, inverse=False, title="Confusion Matrix"):
        fig = plt.figure(title, figsize=(9, 8))
        ax = fig.add_subplot(111)

        top = plt.cm.get_cmap('Oranges_r', 128)
        bottom = plt.cm.get_cmap('Blues', 128)

        newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
        newcmp = ListedColormap(newcolors, name='OrangeBlue')

        if inverse:
            cmap = newcmp
        else:
            cmap = plt.cm.RdYlGn

        im = ax.imshow(
            corr_mat,
            aspect='auto',
            cmap=cmap,
            interpolation='nearest',
        )

        ax.set_xticks(np.arange(len(raw_dict)))
        ax.set_yticks(np.arange(len(raw_dict)))
        ax.set_xticklabels(list(raw_dict.keys()))
        ax.set_yticklabels(list(raw_dict.keys()))

        fig.colorbar(im)

        for i in range(len(raw_dict)):
            for j in range(len(raw_dict)):
                if stdw_mat is not None:
                    text_str = "{0:.1f}\n{1:.1f}".format(corr_mat[i, j], stdw_mat[i, j])
                else:
                    text_str = "{0:.1f}".format(corr_mat[i, j])

                ax.text(j, i, text_str, ha="center", va="center", color="black")

    def _visualize_selected(self, label):
        """
        Visualize selected options.

        Visualize selected options and discard paramter handed
        by event before calling visualize().

        Args:
            label: Label of checked/unchecked option.
        """
        if label == 'reach set':
            self.plot_reach = not self.plot_reach
        elif label == 'frenet trajectories':
            self.plot_frenet = not self.plot_frenet

        self.visualize()


class AttrDict(dict):
    """AttrDict."""

    def __init__(self, *args, **kwargs):
        """Create AttrDict."""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def analyze_multi_correlation(data):
    """Analyze correlation between multiple principles.

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    multi_corr_dict = {}

    df = pd.DataFrame(data=data)

    for key in data:
        reg_str = key + " ~"
        for key2 in data:
            if key != key2:
                reg_str += " + " + key2

        reg_model = ols(reg_str, data=df).fit()
        multi_corr_dict[key] = {
            "r_value": np.sqrt(reg_model.rsquared)
        }
        for idx, val in zip(reg_model.params.index, reg_model.params.values):
            multi_corr_dict[key][idx] = val

    return multi_corr_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()

    logvisualizer = FrenetLogVisualizer(args.logfile, visualize=True)

    # Show correlation matrix of cost terms
    # logvisualizer.multi_correlation()
    # logvisualizer.distance_matrix(plot=True)
    # logvisualizer.correlation_matrix()

    logvisualizer.visualize()
    print("Done.")
