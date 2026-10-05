#!/usr/bin/env python3
"""Compare the custom extension with PyTorch scaled-dot-product attention."""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

import my_flash_attn


def measure(function, warmup, target_ms):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    stop.record()
    stop.synchronize()
    probe = max(start.elapsed_time(stop), 0.005)
    repeats = max(3, min(200, int(target_ms / probe)))
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeats, repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n", type=int, default=1024)
    parser.add_argument("--max-n", type=int, default=16384)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument("--output", default="result/torch_benchmark.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["SequenceLength", "HeadDim", "KernelName", "GFLOPS", "TimeMs", "Repeats"])
        for n in range(args.min_n, args.max_n + 1, args.step):
            if n % 64:
                raise SystemExit("all sequence lengths must be multiples of 64")
            q = torch.randn(n, 64, device="cuda", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            cases = {
                "Custom_V4": lambda: my_flash_attn.run_v4(q, k, v),
                "Custom_V5": lambda: my_flash_attn.run_v5(q, k, v),
                "PyTorch_SDPA": lambda: F.scaled_dot_product_attention(
                    q[None, None], k[None, None], v[None, None]
                ),
            }
            for name, function in cases.items():
                milliseconds, repeats = measure(function, args.warmup, args.target_ms)
                gflops = (4.0 * n * n * 64) / (milliseconds * 1e-3) / 1e9
                writer.writerow([n, 64, name, gflops, milliseconds, repeats])
                file.flush()
                print(f"N={n:6d} {name:14s} {milliseconds:9.3f} ms {gflops:10.2f} GFLOP/s")


if __name__ == "__main__":
    main()
