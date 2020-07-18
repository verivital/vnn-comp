import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlepad'] = 10
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import itertools
import os
from tools.pd2csv import pd2csv


def plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, fig_name, title, create_csv=False):

    mpl.rcParams['font.size'] = 12
    tables = []
    if create_csv and not os.path.exists(folder + "/csv/"):
        os.makedirs(folder + "/csv/")
    for filename in file_list:
        m = pd.read_pickle(folder + filename).dropna(how="all")
        if create_csv:
            pd2csv(folder, folder + "/csv/", filename)
        tables.append(m)

    # keep only the properties in common
    for m in tables:
        m["unique_id"] = m["Idx"].map(str) + "_" + m["prop"].map(str)
    for m1, m2 in itertools.product(tables, tables):
        m1.drop(m1[(~m1['unique_id'].isin(m2["unique_id"]))].index, inplace=True)

    timings = []
    for idx in range(len(tables)):
        timings.append([])
        for i in tables[idx][time_name_list[idx]].values:
            if i >= timeout:
                timings[-1].append(float('inf'))
            else:
                timings[-1].append(i)
        timings[-1].sort()

    # check that they have the same length.
    for m1, m2 in itertools.product(timings, timings):
        assert len(m1) == len(m2)
    print(len(m1))

    starting_point = timings[0][0]
    for timing in timings:
        starting_point = min(starting_point, timing[0])

    fig = plt.figure(figsize=(6, 6))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max + 5])

    min_solve = float('inf')
    max_solve = float('-inf')
    for timing in timings:
        min_solve = min(min_solve, min(timing))
        finite_vals = [val for val in timing if val != float('inf')]
        if len(finite_vals) > 0:
            max_solve = max(max_solve, max([val for val in timing if val != float('inf')]))

    axis_min = starting_point
    axis_min = min(0.5 * min_solve, 1)
    ax_value.set_xlim([axis_min, timeout + 1])

    for idx, (clabel, timing) in enumerate(zip(labels, timings)):
        # Make it an actual cactus plot
        xs = [axis_min]
        ys = [y_min]
        prev_y = 0
        for i, x in enumerate(timing):
            if x <= timeout:
                # Add the point just before the rise
                xs.append(x)
                ys.append(prev_y)
                # Add the new point after the rise, if it's in the plot
                xs.append(x)
                new_y = 100 * (i + 1) / len(timing)
                ys.append(new_y)
                prev_y = new_y
        # Add a point at the end to make the graph go the end
        xs.append(timeout)
        ys.append(prev_y)

        ax_value.plot(xs, ys, color=colors[idx], linestyle='solid', label=clabel, linewidth=3.0)

    ax_value.set_ylabel("% of properties verified", fontsize=15)
    ax_value.set_xlabel("Computation time [s]", fontsize=15)
    plt.xscale('log', nonposx='clip')
    ax_value.legend(fontsize=12)
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()

    figures_path = "./plots/"
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    plt.savefig(figures_path + fig_name, format='pdf', dpi=300)


def plot_vnn_results():

    ## MNIST-eth
    folder = './mnist_results/'
    bases = ["mnist_0.1", "mnist_0.3"]
    for base in bases:
        file_list = [
            f"{base}_KW_prox_100-pinit-eta10.0-feta10.0.pkl",
        ]
        time_base = "BTime_KW"
        time_name_list = [
            f"{time_base}_prox_100",
        ]
        labels = [
            "Proximal BaBSR",
        ]
        timeout = 300
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", f"MNIST ETH: {base}",
                                create_csv=True)

    ## CIFAR-eth
    folder = './cifar_results/'
    bases = ["cifar10_2_255", "cifar10_8_255"]
    for base in bases:
        file_list = [
            # f"{base}_KW_prox_100-pinit-eta1.0-feta1.0.pkl",
            f"{base}_KW_adam_160-pinit-ilr0.0001,flr1e-06.pkl",
            # f"{base}_KW_gurobi.pkl",
            # f"{base}_KW_gurobi-anderson_1.pkl",
        ]
        time_base = "BTime_KW"
        time_name_list = [
            # f"{time_base}_prox_100",
            f"{time_base}_adam_160",
            # f"{time_base}_gurobi",
            # f"{time_base}_gurobi-anderson_1",
        ]
        labels = [
            # "Prox-BaBSR",
            "Supergradient BaBSR",
            # "Gurobi-BaBSR",
            # "Gur-And-BaBSR",
        ]
        timeout = 300
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", f"CIFAR ETH: {base}",
                                create_csv=True)

    folder = './cifar_results/'
    bases = ["base_100", "wide_100", "deep_100"]
    for base in bases:
        file_list = [
            f"{base}_GNN_prox_100-pinit-eta100.0-feta100.0.pkl",
        ]
        time_base = "BTime_GNN"
        time_name_list = [
            f"{time_base}_prox_100",
        ]
        labels = [
            "Proximal BaBGNN",
        ]
        timeout = 3600
        plot_from_tables_custom(folder, file_list, time_name_list, timeout, labels, base + ".pdf", f"CIFAR OVAL: {base}",
                                create_csv=True)

    # plt.show()


if __name__ == "__main__":

    plot_vnn_results()
