# V5 FA2/WMMA design

V5 keeps the numerically stable online-softmax recurrence from earlier versions while changing work partitioning and K/V tile width. It is FA2-style rather than a line-for-line reproduction of the FlashAttention-2 library kernel.

## CTA and warp ownership

- One CTA owns 64 query rows.
- The CTA has 128 threads (four warps).
- Each warp owns 16 query rows for the lifetime of the CTA.
- Q, K, V, and O use FP16 storage; score and output accumulator fragments use FP32.
- K/V tiles contain 64 rows, so a sequence contains `N/64` streaming stages.

This ownership avoids cross-warp reductions for the online-softmax state. Each lane retains two row states (`m`, `l`) according to the Ampere `m16n16k16` accumulator mapping.

## Pipeline

The 40 KiB dynamic shared-memory layout is:

```text
Q tile                         64 × 64 × 2 bytes =  8 KiB
K double buffer          2 × 64 × 64 × 2 bytes = 16 KiB
V double buffer          2 × 64 × 64 × 2 bytes = 16 KiB
                                                     40 KiB
```

Stage 0 is prefetched before the main loop. At each iteration, the CTA submits the next K/V stage with `__pipeline_memcpy_async`, waits until the current read stage is ready, computes QKᵀ and PV with WMMA, then swaps read/write stages. A final `wait_prior(0)` drains the last stage.

For each K/V tile:

1. Four K fragments cover the 64 key positions.
2. WMMA accumulates four `16×16` score fragments per warp.
3. Warp shuffles reduce row maxima and exponential sums.
4. Previous FP32 output fragments are rescaled by `exp(m_old-m_new)`.
5. Half-precision probability fragments multiply V through WMMA and accumulate into FP32 output fragments.
6. After the last tile, each row is divided by its accumulated normalization sum and stored as FP16.

## Supported contract

- NVIDIA Ampere (`sm_80`) or newer; validated on `sm_86`.
- Forward-only self-attention for one head.
- Contiguous FP16 Q/K/V with shape `[N, 64]`.
- `N` is a positive multiple of 64.
- No causal mask, bias, dropout, batching, backward, or ragged sequence support.

The implementation uses the Ampere WMMA accumulator element mapping for row-local online-softmax state. This is architecture-specific code, not a portable CUDA abstraction; extending the compute target requires revalidation.

## Validation evidence

Native regression covers sequence lengths `64, 128, 256, 1024, 4096, 8192` and seeds `1, 7, 42, 2026`. Across those checks, maximum absolute error against the FP32 CPU safe-softmax reference stayed around `1e-4` or below. The committed `make test-v5` target exercises four representative shape/seed pairs and checks all available rows up to 64.

The PyTorch suite runs both V4 and V5 against `torch.nn.functional.scaled_dot_product_attention` and verifies execution on a non-default current CUDA stream.

## Performance interpretation

V5 reduces the number of K/V stages relative to V4 by increasing the tile width from 32 to 64. Historical RTX 3060 Laptop measurements reached 11.53 TFLOP/s at `N=62,464`, versus 10.77 TFLOP/s for V4. The benefit is shape-dependent: at `N=4096`, recent validation runs showed V4 ahead of V5. The project therefore reports the curve rather than claiming that V5 wins at every sequence length.
