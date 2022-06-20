"""Create beautiful correleation boxplots."""

import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np

RAW_DATA_PATHS = [
    './planner/Frenet/RISK_VALUES/ethical.json',
    './planner/Frenet/RISK_VALUES/ego.json',
    './planner/Frenet/RISK_VALUES/standard.json',
]

NO_TOP_VALUES = 100

BLACK = (0, 0, 0)
DARK_GRAY = (0.2, 0.2, 0.2)
TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_BLUE_TRANS20 = (0 / 255, 101 / 255, 189 / 255, 0.2)
TUM_BLUE_TRANS50 = (0 / 255, 101 / 255, 189 / 255, 0.5)
TUM_DARKBLUE = (0 / 255, 82 / 255, 147 / 255)
TUM_LIGHTBLUE = (100 / 255, 160 / 255, 200 / 255)
TUM_ORANGE = (227 / 255, 114 / 255, 34 / 255)
TUM_ORANGE_TRANS50 = (227 / 255, 114 / 255, 34 / 255, 0.5)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
TUM_GREEN_TRANS50 = (162 / 255, 173 / 255, 0 / 255, 0.5)


# Create dictionary of keyword aruments to pass to plt.boxplot
red_dict = {
    'patch_artist': True,
    'boxprops': {"color": DARK_GRAY, "facecolor": TUM_BLUE_TRANS50},
    'capprops': {"color": DARK_GRAY},
    "showfliers": False,
    'flierprops': {"color": TUM_BLUE_TRANS20, "markeredgecolor": TUM_BLUE_TRANS20},
    'medianprops': {"color": DARK_GRAY},
    'whiskerprops': {"color": DARK_GRAY},
}


def colormap(val):
    """Different colors for positive and negative values.

    Args:
        val ([type]): [description]

    Returns:
        [type]: [description]
    """
    if val >= 0:
        return (0 / 255, 101 / 255, 189 / 255, val)

    if val < 0:
        return (227 / 255, 114 / 255, 34 / 255, -val)


def get_x_highest_values(data_dict, x=None):
    """Return the x highest values for every dict.values().

    Args:
        x (_type_): _description_
        data_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_dict = {}

    if x is None:
        x = int(len(list(data_dict.values())[0]) * 0.001)
        print(x)

    for key, val in data_dict.items():
        new_val = sorted(val, reverse=True)[:x]
        new_dict[key] = new_val

    return new_dict


cdict = {
    'red': [[0.0, 0.0, 0.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]],
    'green': [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 1.0, 1.0], [1.0, 1.0, 1.0]],
    'blue': [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]],
}

newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

top = plt.cm.get_cmap('Oranges_r', 128)
bottom = plt.cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')


def box_plot(data, edge_color, fill_color, ax):
    """Generate a boxplot.

    Args:
        data ([type]): [description]
        edge_color ([type]): [description]
        fill_color ([type]): [description]

    Returns:
        [type]: [description]
    """
    bp = ax.boxplot(data.values(), **red_dict)

    for patch, key in zip(bp['boxes'], data.keys()):
        if "EGO" in key:
            patch.set_facecolor(TUM_ORANGE_TRANS50)

        elif "STANDARD" in key:
            patch.set_facecolor(TUM_GREEN_TRANS50)

    return bp


boxplot_data = {}

for p in RAW_DATA_PATHS:
    with open(p) as json_file:
        data = json.load(json_file)

    boxplot_data = {**boxplot_data, **data}

# Group data by road user
sorted_keys = sorted(boxplot_data, key=lambda x: ord(x[-2]))
boxplot_data_sorted = {key: boxplot_data[key] for key in sorted_keys}

fig, ax = plt.subplots()
bp = box_plot(boxplot_data_sorted, TUM_DARKBLUE, TUM_LIGHTBLUE, ax=ax)

ax.set_xticklabels(boxplot_data_sorted.keys(), rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(RAW_DATA_PATHS[0]), "risks_boxplot.pdf"))


# Create boxplot of the x highest risk values each
boxplot_data_high = get_x_highest_values(boxplot_data_sorted, x=NO_TOP_VALUES)

import csv

a_file = open(os.path.join(os.path.dirname(RAW_DATA_PATHS[0]), f"raw_values_top{NO_TOP_VALUES}.csv"), "w")
writer = csv.writer(a_file)
writer.writerow(list(boxplot_data_high.keys()))

values = list(boxplot_data_high.values())
for i in range(len(values[0])):
    # writer.writerow(
    #     [
    #         values[0][i],
    #         values[1][i],
    #         values[2][i],
    #         values[3][i],
    #         values[4][i],
    #         values[5][i],
    #         values[0][i],
    #     ]
    # )
    writer.writerow([values[j][i] for j in range(9)])

a_file.close()


fig2, ax2 = plt.subplots()
bp2 = box_plot(boxplot_data_high, TUM_DARKBLUE, TUM_LIGHTBLUE, ax=ax2)

ax2.set_xticklabels(boxplot_data_high.keys(), rotation=90, fontsize=4)
plt.savefig(
    os.path.join(
        os.path.dirname(RAW_DATA_PATHS[0]), f"risks_boxplot_top{NO_TOP_VALUES}.pdf"
    )
)


plt.tight_layout()
plt.show()

print("Done.")
