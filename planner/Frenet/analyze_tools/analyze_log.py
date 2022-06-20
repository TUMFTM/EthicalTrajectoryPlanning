"""Analyze and visualize logfiles."""

import os
import sys
import pprint
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons
import numpy as np
import argparse
from scipy import stats

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
        self.all_weighted_cost_dict = {}
        weights = self.weights

        for key in weights:
            if weights[key] != 0 and "risk" not in key:
                self.all_weighted_cost_dict[key] = []

        for key in self.all_weighted_cost_dict:
            for data in self._all_log_data:
                (_, _, ft_list_valid, ft_list_invalid, _, _, _) = data

                for traj in ft_list_valid + ft_list_invalid:
                    if key in self.risk_keys:
                        self.all_weighted_cost_dict[key].append(
                            traj["cost_dict"]["risk_cost_dict"][key]
                            * weights[key]
                            * weights["risk_cost"]
                        )
                    else:
                        self.all_weighted_cost_dict[key].append(
                            traj["cost_dict"]["unweighted_cost"][key] * weights[key]
                        )

        corr_mat = np.zeros(
            (len(self.all_weighted_cost_dict), len(self.all_weighted_cost_dict))
        )
        for key1, x in enumerate(self.all_weighted_cost_dict.values()):
            for key2, y in enumerate(self.all_weighted_cost_dict.values()):
                correlation, p_value = stats.pearsonr(x, y)
                corr_mat[key1, key2] = correlation

        if plot:
            self._draw_corr_mat(corr_mat)

        return corr_mat, list(self.all_weighted_cost_dict.keys())

    def _draw_corr_mat(self, corr_mat):
        fig = plt.figure("Correlation Matrix", figsize=(9, 8))
        ax = fig.add_subplot(111)
        im = ax.imshow(
            corr_mat,
            aspect='auto',
            cmap=plt.cm.RdYlGn,
            interpolation='nearest',
        )

        ax.set_xticks(np.arange(len(self.all_weighted_cost_dict)))
        ax.set_yticks(np.arange(len(self.all_weighted_cost_dict)))
        ax.set_xticklabels(list(self.all_weighted_cost_dict.keys()))
        ax.set_yticklabels(list(self.all_weighted_cost_dict.keys()))

        fig.colorbar(im)

        plt.show()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()

    logvisualizer = FrenetLogVisualizer(args.logfile, visualize=True)

    # Show correlation matrix of cost terms
    # logvisualizer.correlation_matrix()

    logvisualizer.visualize()
    print("Done.")
