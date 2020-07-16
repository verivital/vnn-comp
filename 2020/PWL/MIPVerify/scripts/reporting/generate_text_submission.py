#!/usr/bin/env python3
import argparse
import itertools
import os

import pandas as pd


def get_acasxu_header_columns():
    return ["mipverify_time", "mipverify_results", "property_id", "network_id"]


def process_acasxu_result_row(row):
    time = "{0:.2f}".format(row.TotalTime) if row.VerificationResult != "Timeout" else "-"
    return (
        time,
        row.VerificationResult,
        row.PropertyID,
        row.NetworkID.replace("_", "-"),
    )


def get_mnist_oval_header_columns():
    return ["mipverify_time", "mipverify_results", "network_id", "eps", "sample_number"]


def process_mnist_oval_result_row(row):
    time = "{0:.2f}".format(row.TotalTime) if row.VerificationResult != "Timeout" else "-"
    return (
        time,
        row.VerificationResult,
        row.NetworkID,
        row.Eps,
        row.SampleNumber,
    )


def write_output(df, working_directory, process_row, get_header_columns):
    output_path = "{}.txt".format(working_directory)
    pd.DataFrame(
        [process_row(row) for (_, row) in df.iterrows()],
        columns=get_header_columns(),
    ).to_csv(output_path, index=False, sep=" ")


if __name__ == "__main__":
    SUPPORTED_BENCHMARKS = ["ACASXU-ALL", "ACASXU-HARD", "MNIST-OVAL"]

    parser = argparse.ArgumentParser(
        description="Generates .txt files compatible with the main report."
    )
    parser.add_argument("--benchmark_name", choices=SUPPORTED_BENCHMARKS)
    args = parser.parse_args()

    benchmark_name = args.benchmark_name

    working_directory = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../results/{}".format(benchmark_name.lower())
    )
    if benchmark_name == "ACASXU-ALL" or benchmark_name == "ACASXU-HARD":
        results_path = os.path.join(working_directory, "summary.csv")
        df = pd.read_csv(results_path)
        write_output(df, working_directory, process_acasxu_result_row, get_acasxu_header_columns)
    elif benchmark_name == "MNIST-OVAL":
        num_layers_list = [2, 4, 6]
        eps_values = [0.02, 0.05]
        dfs = []
        for (num_layers, eps_value) in itertools.product(num_layers_list, eps_values):
            results_path = os.path.join(
                working_directory,
                "mnist-net_256x{}_linf-norm-bounded-{}".format(num_layers, eps_value),
                "summary.csv",
            )
            df = pd.read_csv(results_path)
            df["NetworkID"] = "MNIST{}".format(num_layers)
            df["Eps"] = eps_value
            dfs.append(df)
        merged_df = pd.concat(dfs)
        write_output(merged_df, working_directory, process_mnist_oval_result_row, get_mnist_oval_header_columns)
    else:
        raise ValueError()
