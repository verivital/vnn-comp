#!/usr/bin/env python3
import argparse
import itertools
import os

import pandas as pd


def get_ggn_cnn_header_columns():
    return ["mipverify_time", "mipverify_results", "network_id", "eps", "sample_number"]


def process_ggn_cnn_result_row(row):
    if row.VerificationResult=="Misclassified":
        return None
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
        filter(lambda x: x is not None, [process_row(row) for (_, row) in df.iterrows()]),
        columns=get_header_columns(),
    ).to_csv(output_path, index=False, sep=" ")


if __name__ == "__main__":
    SUPPORTED_BENCHMARKS = ["GGN-CNN"]

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
    if benchmark_name == "GGN-CNN":
        name_to_params = {
            "mnist_0.1": ("MNIST", "0.1"),
            "mnist_0.3": ("MNIST", "0.3"),
            # "cifar10_2_255": ("CIFAR10", "2/255"),
            # "cifar10_8_255": ("CIFAR10", "8/255"),
        }
        dfs = []
        for (name, params) in name_to_params.items():
            (net_id, eps) = params
            results_path = os.path.join(
                working_directory,
                name,
                "summary.csv",
            )
            df = pd.read_csv(results_path)
            df["NetworkID"] = net_id
            df["Eps"] = eps
            dfs.append(df)
        merged_df = pd.concat(dfs)
        write_output(merged_df, working_directory, process_ggn_cnn_result_row, get_ggn_cnn_header_columns)
    else:
        raise ValueError()
