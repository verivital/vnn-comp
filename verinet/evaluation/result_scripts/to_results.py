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

    outfile_dir = os.path.join(os.path.dirname(__file__), "../results/")

    if not os.path.isdir(outfile_dir):
        os.mkdir(outfile_dir)

    for result_file in result_files:

        statuses, times, _, timeout = read_result_file(os.path.join(results_path, result_file))

        with open(os.path.join(outfile_dir, result_file), "w") as out_file:

            for i, status in enumerate(statuses):

                if status == "Safe":
                    out_file.write("UNSAT\n")
                elif status == "Unsafe":
                    out_file.write("SAT\n")
                else:
                    out_file.write("UKNOWN\n")

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
