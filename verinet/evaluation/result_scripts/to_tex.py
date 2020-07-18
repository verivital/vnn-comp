import os

import numpy as np
import matplotlib.pyplot as plt


# noinspection SpellCheckingInspection
def convert():

    """
    Converts all the results in ../benchmark_results to tex figures and pdf plots.
    """

    results_path = os.path.join(os.path.dirname(__file__), "../benchmark_results/")
    result_files = sorted(os.listdir(results_path))

    outfile_dir = os.path.join(os.path.dirname(__file__), "../benchmark_results_tex/")

    if not os.path.isdir(outfile_dir):
        os.mkdir(outfile_dir)
    if not os.path.isdir(os.path.join(outfile_dir, 'figures')):
        os.mkdir(os.path.join(outfile_dir, 'figures'))
    if not os.path.isdir(os.path.join(outfile_dir, 'tables')):
        os.mkdir(os.path.join(outfile_dir, 'tables'))

    for result_file in result_files:

        outfile = os.path.join(outfile_dir, f"tables/{result_file[:-4]}.tex")
        status, times, epsilons, timeout = read_result_file(os.path.join(results_path, result_file))

        unique_epsilons = np.unique(np.array(epsilons))
        status_sorted_eps = np.array([status[epsilons == ueps] for ueps in unique_epsilons])
        times_sorted_eps = np.array([times[epsilons == ueps] for ueps in unique_epsilons])

        img_nums = []
        for i in range(len(unique_epsilons)):
            img_nums.append(list(range(len(status_sorted_eps[0]))))
        img_nums = np.array(img_nums)

        if status_sorted_eps.shape[1] > 50:

            # To many images, split into multiple columns.

            msg = "ERROR: results with more than one epsilon and more than 50 images" \
                  "or results with more than 100 images not implemented"
            assert status_sorted_eps.shape[0] == 1 and len(status) <= 100, msg

            status_sorted_eps = np.array((status_sorted_eps[0, :50], status_sorted_eps[0, 50:]))
            times_sorted_eps = np.array((times_sorted_eps[0, :50], times_sorted_eps[0, 50:]))
            img_nums = np.array((img_nums[0, :50], img_nums[0, 50:]))
            unique_epsilons = np.array((unique_epsilons[0], unique_epsilons[0]))

        write_table(status_sorted_eps, times_sorted_eps, img_nums, unique_epsilons, result_file[:-4], outfile)

        outfile_path = ''.join([f'{directory}/' for directory in outfile.split('/')[:-2]]) + "figures/"
        outfile = f"{os.path.join(outfile_path, outfile.split('/')[-1][:-4])}.pdf"
        plot(status, times, result_file[:-4], outfile, timeout)


# noinspection PyTypeChecker
def plot(status: np.array[str], times: np.array[float], name: str, outfile: str, timeout: float):

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


def write_table(status: np.array[str, str], times: np.array[float, float], img_idx: np.array[int, int],
                epsilons: np.array[float], name: str, outfile: str):

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
            The solving times corresponding to the status list.
        img_idx:
            The image indices corresponding to the status list.
        epsilons:
            The epsilons corresponding to the status list.
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

        top_tmp = "|llr"
        top = ""
        for i in range(len(epsilons)):
            top += top_tmp
        top += "|"

        f.write(f'  \\begin{{tabular}}{{{top}}}\n')

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


def read_result_file(filename: str) -> tuple:

    """
    Reads the given result file.

    Args:
        filename:
            The file to read.
    Returns:
        statuses, times, epsilons, timeout
    """

    status = []
    times = []
    eps = []
    current_eps = 0
    image = 1

    with open(filename, 'r') as f:

        timeout = float(f.readline().split(' ')[-3])

        for line in f:

            if line.split('=')[0] == "Benchmarking with epsilon ":
                image = 1
                current_eps = float(line.strip().split('=')[1][:-1])

            if line[0:5] != "Final":
                continue

            eps.append(current_eps)

            if line.split(' ')[-2][:-1] == 'predicted':
                times.append(0)
                status.append("Misclassified")
                continue

            time = float(line.split(' ')[-2][:-1])

            if time > timeout:
                times.append(timeout)
                status.append("Undecided")

            else:
                times.append(time)
                status.append(line.split(' ')[5][7:-1])

            image += 1

    return np.array(status), np.array(times), np.array(eps), timeout


if __name__ == '__main__':

    convert()
