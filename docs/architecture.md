# Architecture

## Why fusion matters

Conventional attention materializes `S = QKᵀ / √D` and `P = softmax(S)`. Both are `N×N`, so their memory grows quadratically with sequence length. The fused kernels visit K/V tiles, maintain only the online-softmax state for each query row, and immediately accumulate output contributions.

For a tile with local maximum `m_tile`, exponential sum `l_tile`, and weighted value sum `o_tile`, the running state is updated using a new maximum `m_new`:

```text
alpha = exp(m_old  - m_new)
beta  = exp(m_tile - m_new)
l_new = alpha * l_old + beta * l_tile
o_new = alpha * o_old + beta * o_tile
```

The final output is `o/l`. This stable recurrence is the core invariant shared by the fused forward and decoding paths.

## Data path

1. A query tile is loaded once and retained in shared memory or registers.
2. K/V tiles stream from global memory.
3. Threads or Tensor Cores compute a score tile.
4. Warp reductions produce maxima and normalization sums.
5. The previous output accumulator is rescaled and the new `P·V` contribution is added.
6. Only the normalized output is written to global memory.

Input conversion and allocation are outside the timed native benchmark region. FP16 and FP32 implementations share the same registry and validation harness.

## Optimization stages

- V1 proves the online recurrence and linear-memory execution.
- V2 partitions each query row across four lanes, uses vectorized loads, and combines partial dots with warp shuffles.
- V3 stages alternating K/V buffers with `cp.async` to overlap memory movement and arithmetic.
- V4 maps QKᵀ and PV products to WMMA fragments while retaining online-softmax state in FP32.
- V5 assigns 16 query rows to each warp, doubles the K/V tile width to 64, and overlaps two K/V stages with `cp.async`. It is the stable FA2-style forward path for the current fixed-shape contract.
- Decoding computes split partial `(O, L, M)` states and merges them with the same numerically stable recurrence.

## Stable contract

- Forward only, one attention head per invocation.
- Q/K/V shape `[N, 64]`, contiguous, CUDA, and FP16 for the PyTorch V4 API.
- `N` must be divisible by 64.
- Ampere or newer is required for the asynchronous-copy/WMMA path.
- No causal mask, dropout, bias, batching, backward, or ragged layout yet.

These restrictions are checked at the Python boundary and native CLI rather than being implicit kernel assumptions.
