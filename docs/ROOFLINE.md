# Decode roofline: Qwen2.5-7B bf16 on TPU v6e-1

**v6e spec** ([Cloud TPU v6e docs](https://cloud.google.com/tpu/docs/v6e)): 918 TFLOP/s bf16 (1836 TOPS int8), 32 GB HBM at 1640 GB/s, 256x256 MXU. The ridge point is 918e12 / 1.64e12, about 560 FLOP/byte.

**Model:** 7.62B params, so 15.2 GB in bf16. KV is 2 x 28 layers x 4 KV heads x 128 x 2 B = 56 KiB/token. Context is about 1.4k tokens (mean prompt 1236 plus output).

**Floor per decode step** = max((W + B x ctx x KV) / BW, 2 x P x B / peak). For B <= 32 the memory term dominates by more than 10x. Decode becomes compute-bound only at B around 500, which our HBM can't hold (KV pool fits about 170 sequences).

| c | roofline ITL ms | measured ITL p50 ms | % of roofline | roofline decode tok/s | measured output tok/s |
|---|---|---|---|---|---|
| 1 | 9.34 | 12.88 | 73% | 107 | 75 |
| 4 | 9.49 | 13.59 | 70% | 422 | 276 |
| 8 | 9.68 | 14.48 | 67% | 826 | 515 |
| 16 | 10.08 | 16.39 | 61% | 1588 | 913 |
| 32 | 10.86 | 19.83 | 55% | 2947 | 1515 |

Cache on. Regenerate with `python3 scripts/roofline.py`. Chart: `results/v6e-1-qwen7b/charts/roofline.png`.
End-to-end tok/s is about 51% of the decode-only roofline at c=32. ITL alone is at 55%. The remaining gap is prefill time and request turnover.

**Where the gap likely is (hypotheses, not profiled):**
1. **c=1, 73%.** Fixed per-step overhead of about 3.5 ms: host scheduling, sampling, the 152k-vocab LM head and logits transfer, plus non-fused VPU ops (RMSNorm, RoPE). Weight streaming itself is probably about 80-90% efficient.
2. **Efficiency falls with c (55% at 32).** Every ms of ITL added per sequence is far more than the extra KV bytes explain (+1.5 ms expected vs +7 ms measured). Likely causes: (a) attention with head_dim 128 underfills the 256-wide MXU, and the paged-attention kernel with page 128 is not bandwidth-efficient; (b) chunked prefill (2048) steps mixed into decode steps inflate ITL. The cache-off run (32 ms ITL) shows how strong effect (b) can be.
3. Next step: run XProf on one decode step at c=32 to separate attention kernel time, matmul time and idle/host time.
