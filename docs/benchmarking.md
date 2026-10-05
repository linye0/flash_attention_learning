# Benchmarking protocol

## Native benchmark

The CLI separates machine-readable CSV (stdout or `--output`) from diagnostics and validation (stderr). Inputs use a deterministic seed. Q/K/V allocation, host-to-device copies, and FP32-to-FP16 conversion occur outside timing.

```bash
./build/attention_bench \
  --min-n 1024 --max-n 16384 --step 1024 \
  --kernel V4 --warmup 3 --target-ms 200 \
  --output result/v4.csv
```

One probe launch estimates runtime, then the harness chooses a bounded repeat count for the requested timing window. CUDA events report the arithmetic mean. Approximate forward work is reported as `4*N²*D`; softmax exponentials and reductions are not included, so GFLOP/s is a comparative model rather than an exact instruction count.

## Correctness

The native harness compares selected output rows with an independent FP32 CPU implementation of scaled dot-product attention and safe softmax. Stable FP32 kernels use `1e-3` absolute/relative tolerances; FP16 WMMA paths use `5e-2`.

The PyTorch tests additionally check several sequence lengths, current-stream execution, and rejection of invalid device, dtype, and shape inputs.

## PyTorch comparison

```bash
python3 setup.py build_ext --inplace
python3 benchmarks/benchmark_torch.py --max-n 16384
```

Both implementations receive the same tensors and are timed with CUDA events. The custom API handles a single `[N, D]` head; PyTorch receives the equivalent `[1, 1, N, D]` view.

## Reproducibility

```bash
./scripts/run_benchmark.sh result/runs
```

The run directory records GPU, driver, CUDA compiler, Git revision, CSV results, validation output, and a plot. For defensible results, repeat whole processes, interleave implementation order, report dispersion, and record power/clock state. A laptop display GPU is not an exclusive benchmark environment.

## Nsight Compute

```bash
make profile
```

Relate runtime to Tensor Core/FMA activity, eligible warps, DRAM traffic, shared-memory throughput and bank conflicts, synchronization stalls, and register-limited occupancy. Profiler replay perturbs event timing; use NCU for causal diagnosis and the ordinary harness for latency claims.
