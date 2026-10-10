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
    parser.add_argument("--log-y", action="store_true",
                        help="Use a logarithmic throughput axis for wide performance ranges")
    parser.add_argument("--title", default="FlashAttention Forward Throughput")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    sequence_column = "SequenceLength" if "SequenceLength" in frame else "Seq_Len(N)"
    frame["KernelName"] = frame["KernelName"].str.strip()
    display_names = {
        "V0_Multipass": "V0 Multipass",
        "V1_flash_tiling": "V1 Tiled",
        "V2_flash_vectorized": "V2 Vectorized",
        "V3_flash_pipeline": "V3 Pipeline",
        "V4_flash_wmma": "V4 WMMA",
        "V5_flash_FA2_wmma": "V5 FA2/WMMA",
    }
    frame["KernelName"] = frame["KernelName"].replace(display_names)
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
    axis.set_title(args.title)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Approximate throughput (GFLOP/s)")
    axis.set_xticks(sorted(frame[sequence_column].unique()))
    if args.log_y:
        axis.set_yscale("log")
    axis.legend(title="Implementation", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
