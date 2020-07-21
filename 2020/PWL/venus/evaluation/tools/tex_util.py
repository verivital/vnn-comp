import os

import numpy as np
import matplotlib.pyplot as plt


def plot(status: np.array, times: np.array, name: str, outfile: str, timeout: float):

    """
    Creates and saves the plots of number of cases solved vs time in
    ../benchmark_results_text/figures.

    Args:
        status:
            A list with the verification statuses. Each entry should be either
            "Safe", "Unsafe" or "Undecided".
        times:
            The solving times corresponding to the status list.
        name:
            The name of the network
        outfile:
            The outfile name
        timeout:
            The timeout setting used during verification.
    """

    safe_times = [0] + sorted(times[status == "Safe"]) + [timeout]
    safe_cumsum = [0] + list(np.ones_like(times[status == "Safe"]).cumsum())
    safe_cumsum += [safe_cumsum[-1]]

    unsafe_times = [0] + sorted(times[status == "Unsafe"]) + [timeout]
    unsafe_cumsum = [0] + list(np.ones_like(times[status == "Unsafe"]).cumsum())
    unsafe_cumsum += [unsafe_cumsum[-1]]

    total = [0] + sorted(times[(status == "Safe") + (status == "Unsafe")]) + [timeout]
    total_cumsum = [0] + list(np.ones_like(times[(status == "Safe") + (status == "Unsafe")]).cumsum())
    total_cumsum += [total_cumsum[-1]]

    plt.figure()
    plt.plot(safe_times, safe_cumsum, "g")
    plt.plot(unsafe_times, unsafe_cumsum, "r")
    plt.plot(total, total_cumsum, "b")

    plt.title(name)
    plt.xlabel("Seconds")
    plt.ylabel("Cases Solved")
    plt.legend(["Safe", "Unsafe", "Cases Solved"])

    plt.axis([0.01, timeout, 0, len(status)])

    plt.xscale('log')
    plt.grid()

    plt.savefig(outfile, format="pdf")


def write_table(status: np.array, times: np.array, img_idx: np.array,
                epsilons: np.array, name: str, outfile: str):

    """
    Creates and saves the tex tables containing each individual verification prop.

    The arguments status, times, img_idx should be arrays of the same shape
    as the final latex table. The argument epsilons should contain one epsilon for
    each column.

    Args:
        status:
            A list with the verification statuses of size NxM where N is the number
            of rows and M the number of columns in the latex table.
        times:
            The solving times corresponding to the status list of size NxM.
        img_idx:
            The image indices corresponding to the status list of size NxM.
        epsilons:
            The epsilons corresponding to each column in status of size M.
        name:
            The name of the network.
        outfile:
            The path of the outfile.
    """

    with open(outfile, "w") as f:

        f.write('\\begin{table}[!ht]\n')
        f.write('  \\centering\n')
        f.write(f'  \\caption{{{name.replace("_", " ")}}}\n')
        f.write('  \\footnotesize\n')

        col_format = ""
        for i in range(len(epsilons)):
            col_format += "|llr"
        col_format += "|"

        f.write(f'  \\begin{{tabular}}{{{col_format}}}\n')

        f.write('    \\toprule\n')

        eps_str = "    "
        for eps in epsilons:
            eps_str += f"\\multicolumn{{3}}{{|c|}}{{Epsilon: {eps:.3f}}} & "
        f.write(eps_str[:-2]+"\\\\\n")

        f.write('    \\midrule\n')

        top_bar = "    "

        for j in range(len(epsilons)):
            top_bar += "Image & Status & Time(s) &"
        top_bar = top_bar[:-2] + "\\\\ \n"
        f.write(top_bar)

        f.write('    \\midrule\n')

        for i in range(len(img_idx[0])):
            column_str = f""
            for j in range(len(epsilons)):
                if len(status[j]) > i:
                    column_str += f"    {img_idx[j][i]} & {status[j][i]} & {times[j][i]} &"
            column_str = column_str[:-2] + "\\\\ \n"
            f.write(column_str)

        f.write('    \\bottomrule\n')
        f.write('  \\end{tabular}\n')
        f.write('\\end{table}\n\n')
