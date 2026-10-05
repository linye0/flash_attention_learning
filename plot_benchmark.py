#!/usr/bin/env python3
"""Plot throughput from attention_bench or the PyTorch benchmark."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="result/benchmark_data.csv")
    parser.add_argument("output", nargs="?", default="result/performance_comparison.png")
    parser.add_argument("--kernel", action="append", default=[])
    args = parser.parse_args()

    frame = pd.read_csv(args.input, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    sequence_column = "SequenceLength" if "SequenceLength" in frame else "Seq_Len(N)"
    frame["KernelName"] = frame["KernelName"].str.strip()
    frame["GFLOPS"] = pd.to_numeric(frame["GFLOPS"], errors="coerce")
    frame[sequence_column] = pd.to_numeric(frame[sequence_column], errors="coerce")
    frame = frame.dropna(subset=[sequence_column, "GFLOPS"])
    if args.kernel:
        frame = frame[frame["KernelName"].str.contains("|".join(args.kernel), case=False, na=False)]
    if frame.empty:
        raise SystemExit("No benchmark rows matched")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=frame, x=sequence_column, y="GFLOPS", hue="KernelName",
                 style="KernelName", markers=True, dashes=False, linewidth=2.2, ax=axis)
    axis.set_title("FlashAttention Forward Throughput")
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Approximate throughput (GFLOP/s)")
    axis.legend(title="Implementation", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
