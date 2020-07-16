#!/usr/bin/env python3
import argparse
import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from pylatex import Document, Section, Subsection, Subsubsection, LongTable, \
    Table, Figure, Label, NoEscape, Ref
from pylatex.utils import verbatim, bold

sns.set(
    context="talk",
    style="whitegrid",
    rc={"xtick.bottom": True, "ytick.left": True}
)
plt.rcParams["figure.figsize"] = (10, 8)


def generate_cactus_plot(df, timeout, description, time_column="TotalTime"):
    plt.step(
        range(len(df)+1),
        [0]+sorted(df[time_column]),
        where="post"
    )

    plt.xlim(left=0, right=len(df))
    plt.ylim(bottom=0.01, top=timeout)

    plt.yscale("log")

    plt.xlabel("Instances solved")
    plt.ylabel("Time / s")
    plt.suptitle("Cactus plot for MIPVerify on {}".format(description, timeout))


def add_cactus_plot(doc, df, benchmark_name, timeout_secs):
    if "TotalTime" not in df.columns:
        raise ValueError()

    with doc.create(Figure(position="htb")) as plot:
        generate_cactus_plot(df, timeout_secs, benchmark_name)
        plot.add_plot(width=NoEscape(r"\textwidth"), dpi=300)
        plot.add_caption(
            "Cactus plot for MIPVerify on {}, with timeout at {} seconds.".format(benchmark_name, timeout_secs)
        )


def add_summary_table(doc, df, benchmark_name, timeout_secs):
    if "VerificationResult" not in df.columns:
        raise ValueError()

    doc.append("We present a summary of results in Table ")
    doc.append(Ref("tab:mipverify/{}-summary".format(benchmark_name)))
    doc.append(".")

    with doc.create(Table(position="!h")) as table:
        table.add_caption("Summary of results for MIPVerify on {}. Timeouts were set at {} secs.".format(benchmark_name, timeout_secs))
        table.append(Label("tab:mipverify/{}-summary".format(benchmark_name)))
        with doc.create(LongTable("l|r", booktabs=True)) as t:
            t.add_row((bold("Result"), bold("Count")))
            t.add_hline()
            for (result, count) in df.VerificationResult.value_counts().iteritems():
                summary_result_to_report_result = {
                    "Timeout": verbatim("TIMEOUT"),
                    "SAT": verbatim("UNSAFE"),
                    "UNSAT": verbatim("SAFE"),
                }
                t.add_row((summary_result_to_report_result[result], count))


def add_detail_table(doc, df, table_spec, header_row, row_process_f):
    """
    Params
    ------
    df : pd.DataFrame
    table_spec : str
        A string that represents how many columns a table should have and if it
        should contain vertical lines and where.
    header_row : Array[str]
        Text in header row.
    row_process_f : Callable[pd.Series, Array[Any]]
        Converts each row in results dataframe into an entry in the detail table.
    """
    with doc.create(LongTable(table_spec, booktabs=True)) as t:
        t.add_row(tuple(map(bold, header_row)))
        t.add_hline()
        t.end_table_header()
        for (_, row) in df.iterrows():
            t.add_row(row_process_f(row))


def process_acasxu_result_row(row):
    summary_result_to_report_result = {
        "Timeout": "—",
        "SAT": verbatim("UNSAFE"),
        "UNSAT": verbatim("SAFE"),
    }
    summary_result = row.VerificationResult
    time = "{0:.2f}".format(row.TotalTime) if summary_result != "Timeout" else "—"
    return (
        row.PropertyID,
        row.NetworkID.replace("_", "-"),
        summary_result_to_report_result[summary_result],
        time,
    )


def get_acasxu_report_doc(df, benchmark_name, timeout_secs):
    doc = Document()
    with doc.create(Section("PWL")):
        with doc.create(Subsection(benchmark_name)):
            with doc.create(Subsubsection("MIPVerify")):
                add_cactus_plot(doc, df, benchmark_name, timeout_secs)
                add_summary_table(doc, df, benchmark_name, timeout_secs)
                add_detail_table(
                    doc,
                    df,
                    "cll|r",
                    ["Prop", "Net", "Result", "Runtime / s"],
                    process_acasxu_result_row,
                )
    return doc


if __name__ == "__main__":
    SUPPORTED_BENCHMARKS = ["ACASXU-ALL", "ACASXU-HARD"]

    parser = argparse.ArgumentParser(
        description="Generates summary `.tex` and `.pdf` files in the same directory as the results generated."
    )
    parser.add_argument("--benchmark_name", choices=SUPPORTED_BENCHMARKS)
    args = parser.parse_args()

    benchmark_name = args.benchmark_name

    if benchmark_name == "ACASXU-ALL":
        working_directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../results/acasxu-all"
        )

        def get_doc(df):
            return get_acasxu_report_doc(df, benchmark_name, 5*60)
    elif benchmark_name == "ACASXU-HARD":
        working_directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../results/acasxu-hard"
        )

        def get_doc(df):
            return get_acasxu_report_doc(df, benchmark_name, 6*60*60)
    else:
        raise ValueError()


    results_path = os.path.join(working_directory, "summary.csv")
    doc = get_doc(pd.read_csv(results_path))
    report_path = os.path.join(working_directory)
    doc.generate_pdf(report_path, clean_tex=False)
