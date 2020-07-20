import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from tools.plot_utils import custom_plot


def time_vs_bounds_plot():
    # Plot time vs. bound plots, assumes pickle structure of anderson_bound_per_time

    folder = "./anderson_results/time_vs_bounds/"
    bigm_prox_file = folder + "expLP-cpu-bigmprox-eta1-ACAS-prop1-1_1_data.pth" #"expLP-cpu-bigmprox-etasmaller-ACAS-prop1-1_1_data.pth"  #"expLP-cpu-bigmprox-eta1-ACAS-prop1-1_1_data.pth"
    gurobi_pla_file = folder + "gurobi-planet-6threads-ACAS-prop1-1_1_data.pth"
    proxlp_file = folder + "proxLP-cpu-ACAS-prop1-1_1_data.pth" #"proxLP-cpu-naiveinit-ACAS-prop1-1_1_data.pth"
    bigm_subg_file = folder + "expLP-cpu-bigm-subgradient-ACAS-prop1-1_1_data.pth"

    bigm_prox_measurements = torch.load(bigm_prox_file)
    proxlp_measurements = torch.load(proxlp_file)
    gurobi_pla_measurements = torch.load(gurobi_pla_file)
    bigm_subg_measurements = torch.load(bigm_subg_file)

    bigmprox_lbs = bigm_prox_measurements['lbs']
    bigmprox_time = bigm_prox_measurements['times']
    bigmprox_ubs = bigm_prox_measurements['ubs']
    n_bigmprox_lines = len(bigmprox_lbs)
    n_proxlp_lines = len(proxlp_measurements['lbs'])

    ylog = False  # bounds can be negative, so bad idea (don't want to rescale)
    xlog = True

    n_gurobi_pla_threads = list(gurobi_pla_measurements['lbs'].keys())[0]
    plt.figure(0)
    plt.axvline(gurobi_pla_measurements['times'][n_gurobi_pla_threads], color=colors[n_bigmprox_lines + n_proxlp_lines + 1],
              label="Gurobi PLANET", dashes=(5,5), lw=1)
    plt.axhline(gurobi_pla_measurements['lbs'][n_gurobi_pla_threads], color=colors[n_bigmprox_lines + n_proxlp_lines + 1],
              dashes=(5,5), lw=1)
    plt.subplots_adjust(left=0.17)
    plt.figure(1)
    plt.axvline(gurobi_pla_measurements['times'][n_gurobi_pla_threads], color=colors[n_bigmprox_lines + n_proxlp_lines + 1],
                label="Gurobi PLANET", dashes=(5, 5), lw=1)
    plt.axhline(gurobi_pla_measurements['ubs'][n_gurobi_pla_threads], color=colors[n_bigmprox_lines + n_proxlp_lines + 1],
                dashes=(5, 5), lw=1)
    plt.subplots_adjust(left=0.15)

    for id, key in enumerate(bigmprox_lbs):
        custom_plot(0, bigmprox_time[key], bigmprox_lbs[key], None, "Time [s]", "Lower Bound", "Lower bound vs time", errorbars=False,
                    labelname=r"big-m Prox $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[id])
        custom_plot(1, bigmprox_time[key], bigmprox_ubs[key], None, "Time [s]", "Upper Bound", "Upper bound vs time",
                    errorbars=False,
                    labelname=r"big-m Prox $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[id])

    for id, key in enumerate(proxlp_measurements['lbs']):
        custom_plot(0, proxlp_measurements['times'][key], proxlp_measurements['lbs'][key], None, "Time [s]", "Lower Bound", "Lower bound vs time",
                    errorbars=False,
                    labelname=r"ProxLP $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[n_bigmprox_lines + id])
        custom_plot(1, proxlp_measurements['times'][key], proxlp_measurements['ubs'][key], None, "Time [s]", "Upper Bound", "Upper bound vs time",
                    errorbars=False,
                    labelname=r"ProxLP $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[n_bigmprox_lines + id])

    for id, key in enumerate(bigm_subg_measurements['lbs']):
        custom_plot(0, bigm_subg_measurements['times'][key], bigm_subg_measurements['lbs'][key], None, "Time [s]",
                    "Lower Bound", "Lower bound vs time", errorbars=False, labelname="big-M adam", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[n_bigmprox_lines + n_proxlp_lines + 2 + id])
        custom_plot(1, bigm_subg_measurements['times'][key], bigm_subg_measurements['ubs'][key], None, "Time [s]",
                    "Upper Bound", "Upper bound vs time", errorbars=False, labelname="big-M adam", dotted="-", xlog=xlog,
                    ylog=ylog, color=colors[n_bigmprox_lines + n_proxlp_lines + 2 + id])

    # this is a bit faster than KW init this context
    # proxlp_file = folder + "proxLP-cpu-naiveinit-ACAS-prop1-1_1_data.pth"
    # proxlp_measurements = torch.load(proxlp_file)
    # for id, key in enumerate(proxlp_measurements['lbs']):
    #     custom_plot(0, proxlp_measurements['times'][key], proxlp_measurements['lbs'][key], None, "Time [s]", "Lower Bound", "Lower bound vs time",
    #                 errorbars=False,
    #                 labelname=r"Naive init ProxLP $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
    #                 ylog=ylog, color=colors[-1])
    #     custom_plot(1, proxlp_measurements['times'][key], proxlp_measurements['ubs'][key], None, "Time [s]", "Upper Bound", "Upper bound vs time",
    #                 errorbars=False,
    #                 labelname=r"Naive init ProxLP $\eta=$" + f"{key[0]}" + r", $it_{inner}=$" + f"{key[1]}", dotted="-", xlog=xlog,
    #                 ylog=ylog, color=colors[-1])

    # this is terribly slow in this context
    # proxlp_subg_file = folder + "proxLP-cpu-subgradient-ACAS-prop1-1_1_data.pth"  # "proxLP-cpu-vanillasubg-ACAS-prop1-1_1_data.pth"
    # proxlp_subg_measurements = torch.load(proxlp_subg_file)
    # for id, key in enumerate(proxlp_subg_measurements['lbs']):
    #     custom_plot(0, proxlp_subg_measurements['times'][key], proxlp_subg_measurements['lbs'][key], None, "Time [s]",
    #                 "Lower Bound", "Lower bound vs time", errorbars=False, labelname=r"ProxLP adam",
    #                 dotted="-", xlog=xlog, ylog=ylog, color=colors[-2])
    #     custom_plot(1, proxlp_subg_measurements['times'][key], proxlp_subg_measurements['ubs'][key], None, "Time [s]",
    #                 "Upper Bound", "Upper bound vs time", errorbars=False, labelname=r"ProxLP adam",
    #                 dotted="-", xlog=xlog, ylog=ylog, color=colors[-2])


def adam_cifar_comparison():
    # old planet adam vs big-m adam. Needs to be rerun after the adam fix (big-m will be beaten)

    img_idx = 0 #50  # 0
    step_size = 1e-3

    pickle_name = "../results/icml20/timings-img{}-{},stepsize:{}.pickle"

    planet = torch.load(pickle_name.format(img_idx, "planet-adam", step_size), map_location=torch.device('cpu'))
    custom_plot(0, planet.get_last_layer_time_trace(), planet.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"Planet ADAM $\alpha=$" + f"{step_size}", dotted="-", xlog=False,
                ylog=False, color=colors[0])

    bigm = torch.load(pickle_name.format(img_idx, "bigm-adam", step_size), map_location=torch.device('cpu'))
    custom_plot(0, bigm.get_last_layer_time_trace(), bigm.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"Big-M ADAM $\alpha=$" + f"{step_size}", dotted="-", xlog=False,
                ylog=False, color=colors[1])


def adam_vs_prox():
    # adam bounds vs prox bounds over time (KW initialization)

    img = 10

    folder = "../../Mini-Projects/convex-relaxations/results/icml20/"
    adam_name = folder + f"timings-img{img}-planet-adam,istepsize:0.01,fstepsize1e-06.pickle"
    prox_name = folder + f"timings-img{img}-proxlp,eta:100.0-feta100.0-mom0.0.pickle"

    momentum = 0.4
    optimized_prox_name = folder + f"timings-img{img}-proxlp,eta:1000.0-feta1000.0-mom{momentum}.pickle"

    adam = torch.load(adam_name, map_location=torch.device('cpu'))
    custom_plot(0, adam.get_last_layer_time_trace(), adam.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"ADAM $\alpha \in$" + "[1e-3, 1e-6]", dotted="-", xlog=False,
                ylog=False, color=colors[0])

    prox = torch.load(prox_name, map_location=torch.device('cpu'))
    custom_plot(0, prox.get_last_layer_time_trace(), prox.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"GS Prox $\eta=$" + f"1e2", dotted="-", xlog=False,
                ylog=False, color=colors[1])

    optimized_prox = torch.load(optimized_prox_name, map_location=torch.device('cpu'))
    custom_plot(0, optimized_prox.get_last_layer_time_trace(),
                optimized_prox.get_last_layer_bounds_means_trace(first_half_only_as_ub=True),
                None, "Time [s]", "Upper Bound", "Upper bound vs time", errorbars=False,
                labelname=r"Momentum Prox $\eta=$" + f"1e2", dotted="-", xlog=False,
                ylog=False, color=colors[2])


def five_plots_comparison():

    define_linear_approximation = False

    # images = [0, 5, 10, 15, 20]
    adam_algorithms = [
        # algo, beta1, inlr, finlr
        # ("planet-adam", 0.9, 1e-2, 1e-4),
        # ("planet-adam", 0.9, 1e-3, 1e-6),
        ("planet-adam", 0.9, 1e-4, 1e-6),# Best for cifar10_2_255, cifar10_8_255
        # ("planet-adam", 0.9, 1e-3, 1e-4),
        # ("planet-adam", 0.9, 1e-2, 1e-2),
        # ("planet-adam", 0.9, 1e-3, 1e-3),
        # ("planet-adam", 0.9, 1e-4, 1e-4),
        # ("planet-adam", 0.9, 1e-1, 1e-4),
        # ("planet-adam", 0.9, 1e-1, 1e-3),  # Best for mnist
    ]
    prox_algorithms = [  # momentum tuning
        # algo, momentum, ineta, fineta
        # ("proxlp", 0.0, 1e2, 1e2),
        # ("proxlp", 0.0, 1e1, 1e1),  # Best for mnist
        # ("proxlp", 0.0, 1e3, 1e3),
        ("proxlp", 0.0, 1e0, 1e0),  # Best for cifar10_2_255 and cifar10_8_255
        # ("proxlp", 0.0, 1e4, 1e4),
        # ("proxlp", 0.0, 5e1, 1e2),
        # ("proxlp", 0.0, 5e2, 1e3),
        # ("proxlp", 0.0, 5e0, 1e1),
        # ("proxlp", 0.0, 5e3, 1e4),
    ]

    # folder = "./timings_cifar/"

    ##### mnist-eth #################
    net = "mnist_0.3"  # mnist_0.1 or mnist_0.3 
    folder = f"./timings_mnist/{net}_"
    # read images from file
    try:
        with open(f"./data/undecided-mnist-eps{net[6:]}.txt", "r") as file:
            line = file.readline()
        images = line.split(", ")[:-1]
        images = [int(img) for img in images]
        if len(images) > 5:
            images = images[:5]
    except FileNotFoundError:
        print("No undecided images for the dataset.")
    ##################################

    ##### cifar10-eth #################
    net = "cifar10_8_255"   # cifar10_8_255 or cifar10_2_255
    folder = f"./timings_cifar10/{net}_"
    images = list(range(19)) # all 100 images for workshop
    ##################################

    algorithm_name_dict = {
        "planet-adam": "ADAM",
        "planet-auto-adagrad": "AdaGrad",
        "planet-auto-adam": "Autograd's ADAM",
        "proxlp": "Proximal",
        "jacobi-proxlp": "Jacobi Proximal",
        "dj-adam": "Dvijotham ADAM"
    }

    lin_approx_string = "" if not define_linear_approximation else "-allbounds"

    fig_idx = 0
    for img in images:
        color_id = 0
        for algo, beta1, inlr, finlr in adam_algorithms:
            adam_name = folder + f"timings-img{img}-{algo},istepsize:{inlr},fstepsize:{finlr},beta1:{beta1}{lin_approx_string}.pickle"
            nomomentum = " w/o momentum" if beta1 == 0 else ""

            try:
                adam = torch.load(adam_name, map_location=torch.device('cpu'))
            except:
                continue
            custom_plot(fig_idx, adam.get_last_layer_time_trace(), adam.get_last_layer_bounds_means_trace(second_half_only=True),
                        None, "Time [s]", "Lower Bound", "Lower bound vs time", errorbars=False,
                        labelname=rf"{algorithm_name_dict[algo]} $\alpha \in$" + f"[{inlr}, {finlr}]" + nomomentum,
                        dotted="-", xlog=False,
                        ylog=False, color=colors[color_id])
            color_id += 1

        for algo, momentum, ineta, fineta in prox_algorithms:

            acceleration_string = ""
            if algo != "jacobi-proxlp":
                acceleration_string += f"-mom:{momentum}"
            prox_name = folder + f"timings-img{img}-{algo},eta:{ineta}-feta:{fineta}{acceleration_string}{lin_approx_string}.pickle"

            acceleration_label = ""
            if momentum:
                acceleration_label += f"momentum {momentum}"

            try:
                prox = torch.load(prox_name, map_location=torch.device('cpu'))
            except:
                continue
            custom_plot(fig_idx, prox.get_last_layer_time_trace(), prox.get_last_layer_bounds_means_trace(second_half_only=True),
                        None, "Time [s]", "Lower Bound", "Lower bound vs time", errorbars=False,
                        labelname=rf"{algorithm_name_dict[algo]} $\eta \in$" + f"[{ineta}, {fineta}], " +
                                  f"{acceleration_label}",
                        dotted="-", xlog=False, ylog=False, color=colors[color_id])
            color_id += 1
        fig_idx += 1
        plt.savefig('%d.png'%img)
        plt.show()





if __name__ == "__main__":

    # time_vs_bounds_plot()
    # adam_cifar_comparison()
    # adam_vs_prox()
    five_plots_comparison()

    plt.show()
