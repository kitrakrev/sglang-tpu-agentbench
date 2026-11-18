# DRAFT, not posted: SGLang on TPU v6e, multi-turn agent serving performance

**TL;DR:** On one Cloud TPU v6e chip, Qwen2.5-7B-Instruct served by SGLang (sgl_jax) with the radix prefix cache on
does **1,515 output tok/s at 32 concurrent agent sessions vs 930 tok/s with it off (+63%)**, and cuts **TTFT p99 from
598 ms to 134 ms (4.5x)**. 84-88% of prompt tokens are served from cache. Median TTFT stays flat at ~40 ms across turns.

## Why agents are different
Agent sessions resend a long shared system prompt (tool schemas) and append a tool result every turn,
so the prompt grows each turn. Chat benchmarks (one-shot random prompts) miss this. The question is
whether the radix prefix cache keeps per-turn latency flat on TPU.

## Setup
- Hardware: Cloud TPU v6e-1 (1 chip, 31.24 GiB HBM), us-east5-b, on-demand (spot attempt was preempted after ~5 min).
- Model: Qwen/Qwen2.5-7B-Instruct, bf16, TP=1 (ungated). v6e-8 + Qwen2.5-72B (TP=8): see v6e-8 section.
- Server: sgl_jax (sgl-project/sglang-jax, main @ 2026-11-18), `--attention-backend=fa --page-size=128
  --chunked-prefill-size=2048 --mem-fraction-static=0.85 --max-running-requests=128 --enable-cache-report`;
  cache-off run adds `--disable-radix-cache`. KV pool: 231,552 tokens (12.4 GB).
- Workload: 4 sessions per concurrency level x 6 turns (c x 4 sessions x 6 = 24..768 requests), 600-word shared
  system prompt, 150-word tool results, 128 max output tokens, temperature 0. Mean prompt ~1,240 tokens.
  Concurrency 1/4/8/16/32. Harness: github.com/kitrakrev/sglang-tpu-agentbench (agentbench).
- Cost basis: $2.70/chip-h on-demand (what was billed), $1.00/chip-h spot (estimate).

## Results: v6e-1, Qwen2.5-7B
| cache | c | out tok/s | req/s | TTFT p50/p90/p99 ms | ITL p50/p90/p99 ms | E2E p50/p99 ms | cached % | TTFT t1 / t6 ms | $/1M out (OD / spot) |
|---|---|---|---|---|---|---|---|---|---|
| on | 1 | 75.0 | 1.26 | 30.2 / 31.5 / 32.7 | 12.88 / 12.91 / 12.92 | 392 / 1674 | 84.1 | 28.5 / 30.5 | 10.01 / 3.71 |
| on | 4 | 275.9 | 3.96 | 39.5 / 40.6 / 44.0 | 13.59 / 13.96 / 14.74 | 562 / 1826 | 85.5 | 29.8 / 40.2 | 2.72 / 1.01 |
| on | 8 | 514.7 | 7.45 | 39.8 / 51.8 / 64.6 | 14.48 / 14.90 / 15.36 | 581 / 1961 | 87.7 | 45.1 / 40.2 | 1.46 / 0.54 |
| on | 16 | 913.4 | 14.11 | 40.6 / 76.7 / 98.6 | 16.39 / 17.13 / 18.38 | 563 / 2195 | 87.2 | 76.4 / 41.2 | 0.82 / 0.30 |
| on | 32 | **1515.2** | 24.45 | 42.1 / 95.4 / **134.1** | 19.83 / 21.01 / 22.51 | 650 / 2682 | 87.6 | 83.2 / 42.2 | **0.50 / 0.18** |
| off | 1 | 72.4 | 1.22 | 63.5 / 67.1 / 80.8 | 12.86 / 12.91 / 12.94 | 426 / 1725 | 0 | 42.5 / 66.6 | 10.36 / 3.84 |
| off | 4 | 253.6 | 3.60 | 73.6 / 94.5 / 113.6 | 14.56 / 15.58 / 16.62 | 635 / 2076 | 0 | 82.3 / 74.6 | 2.96 / 1.10 |
| off | 8 | 433.5 | 6.21 | 74.0 / 128.2 / 186.5 | 16.93 / 18.65 / 20.33 | 785 / 2437 | 0 | 128.1 / 75.3 | 1.73 / 0.64 |
| off | 16 | 668.1 | 10.36 | 76.5 / 198.0 / 340.9 | 22.28 / 25.24 / 28.33 | 861 / 3190 | 0 | 195.3 / 76.9 | 1.12 / 0.42 |
| off | 32 | 930.0 | 15.27 | 109.2 / 358.5 / 598.1 | 32.39 / 37.01 / 41.58 | 1208 / 4684 | 0 | 326.5 / 108.3 | 0.81 / 0.30 |

Error rate 0% on all 1,920 requests. Tokens/s per chip = out tok/s (1 chip). HBM: 27.08 / 31.24 GiB after load
(weights + pre-allocated KV pool), 27.09-27.18 GiB after the sweep. Server startup 40 s incl. XLA precompile of
5 EXTEND + 8 DECODE buckets (~17 s, with a warm `JAX_COMPILATION_CACHE_DIR`); post-start warmup 1-2 s.
Cold (empty cache) compile time was not captured cleanly (first run was restarted).

Charts: `results/v6e-1-qwen7b/charts/` (output_tok_per_s, ttft_p50_ms, ttft_p99_ms, itl_p50_ms, e2e_p99_ms, ttft_by_turn).

![tok/s](../results/v6e-1-qwen7b/charts/output_tok_per_s.png)
![TTFT p99](../results/v6e-1-qwen7b/charts/ttft_p99_ms.png)
![TTFT by turn](../results/v6e-1-qwen7b/charts/ttft_by_turn.png)

## Findings
1. Prefix caching matters more as concurrency grows: +4% tok/s at c=1, +63% at c=32; TTFT p99 2.5x lower at c=1, 4.5x at c=32.
   With cache off, prefill of ~1.2k tokens per request competes with decode and inflates ITL p50 (32 vs 20 ms at c=32).
2. The cache hit rate is 84-88% from turn 1, because the 600-word system prompt is shared by every session. Later turns
   also reuse each session's own history.
3. Turn-1 TTFT at high concurrency is dominated by the burst of sessions starting together (cache off: 327 ms turn 1 vs
   108 ms turn 6 at c=32), not by prompt length. That makes "turn 1 / turn N" a queueing metric here, not a cache metric.
4. Cost: $0.50 per 1M output tokens at on-demand price, ~$0.18 at spot, at c=32 with cache on.
5. Gotchas: v6e spot capacity was unavailable in us-east5-b and us-east1-d; the europe-west4-a spot VM was preempted
   within minutes. XLA precompile runs at server start (keep JAX_COMPILATION_CACHE_DIR between restarts). sgl_jax
   accepts `--enable-cache-report` and `--disable-radix-cache` unchanged.

## How this maps to the TPU optimization guide
Google's [TPU Model Design Considerations and Optimization Guide](https://ai-hypercomputer.github.io/tpu-codesign/) (successor to the JAX Scaling Book) explains these results:
- **Decode is HBM-bound.** Per-token latency can't go below weight bytes / HBM BW: 15.2 GB / 1.64 TB/s = 9.3 ms on v6e-1. bf16 decode stays memory-bound until batch is about 500, so higher concurrency is the cheapest lever.
- **Roofline:** at c=1, ITL 12.9 ms is 73% of the floor. At c=32, ITL 19.8 ms is 55% of the floor (2,950 tok/s decode roofline vs 1,515 measured end to end). See [ROOFLINE.md](ROOFLINE.md).
- **Prefix caching** skips prefill FLOPs, not decode KV reads. That is why it cuts TTFT 4.5x but leaves ITL p50 at 20 ms.
- **Shape alignment:** Qwen2.5-7B hidden/FFN dims are 256-aligned, but head_dim 128 underfills v6e's 256x256 MXU in attention.
- **Next steps** from the guide: c=64-128, KV page 256, chunked prefill sweep, int8 weights/KV ([NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md)).

## Run A (2026-11-18): v6e-1 Qwen2.5-7B tuning (spot, europe-west4-a, ~25 min VM time)
All cache on, 0 errors on 7,680 requests. Full table: `results/v6e-1-qwen7b-tuning/metrics.md`; roofline: `roofline.md`.

| variant | c | out tok/s | TTFT p50/p99 ms | ITL p50/p99 ms | cached % | SLO % (TTFT<200, TPOT<25) | goodput tok/s | $/1M out OD / spot | % roofline (ITL) |
|---|---|---|---|---|---|---|---|---|---|
| max-running 128 | 64 | 2,084 | 64.6 / 421.7 | 26.6 / 32.9 | 83.7 | 26.4 | 616 | 0.36 / 0.13 | 47% |
| max-running 128 | **96** | **2,755** | 70.1 / 520.1 | 30.1 / 35.7 | 87.9 | 7.4 | 301 | **0.27 / 0.10** | 46% |
| max-running 128 | 128 | 2,463 | 683.9 / 1,807 | 32.1 / 60.3 | 74.8 | 1.4 | 67 | 0.31 / 0.11 | 48% |
| page 128 | 32 | **1,523** | 42.0 / 138.4 | 19.7 / 22.7 | 87.2 | 100 | **1,523** | 0.49 / 0.18 | 55% |
| page 256 | 32 | 1,471 | 43.6 / 228.0 | 20.2 / 23.1 | 81.4 | 96.9 | 1,449 | 0.51 / 0.19 | - |
| chunk 512 | 32 | 1,479 | 41.4 / 189.1 | 20.4 / 23.2 | 83.8 | 99.5 | 1,467 | 0.51 / 0.19 | - |
| chunk 1024 | 32 | 1,504 | 42.3 / 145.8 | 20.0 / 22.6 | 83.8 | 99.6 | 1,499 | 0.50 / 0.19 | - |
| chunk 2048 | 32 | 1,493 | 42.1 / 135.1 | 20.2 / 22.9 | 83.7 | 100 | 1,494 | 0.50 / 0.19 | - |
| chunk 4096 | 32 | 1,496 | 41.9 / 181.2 | 20.2 / 22.7 | 83.7 | 99.9 | 1,492 | 0.50 / 0.19 | - |

Findings:
1. Raw throughput peaks at c=96 (2,755 tok/s, 1.8x c=32) and falls at c=128: TTFT p50 jumps 10x (684 ms) and the
   cache hit rate drops to 75%, consistent with the KV pool (~170 seqs) evicting prefixes. ITL stays 46-48% of the HBM floor.
2. Under the per-request goodput SLO (TTFT<200 ms and mean ITL<25 ms) c=32 is still the best operating point: c>=64 breaks TPOT 25 ms.
   Pick c=96 for batch/offline agents (cheapest $/token), c=32 for interactive.
3. Page size 256 is not better: slightly lower hit rate (81 vs 87%) from coarser prefix granularity and worse TTFT p99. Keep 128.
4. Chunked prefill 512-4096 is flat on throughput (+-1.5%); 1024-2048 give the best TTFT p99. Default 2048 is fine.
5. XProf trace of 20 decode steps at c=32 captured (`profile_c32/trace.tgz`, 454 MB, kept out of git, local only).

## v6e-8, Qwen2.5-72B
Not run yet. On 2026-11-18 ~09:50-10:05 PST, v6e-8 spot had no capacity in us-east5-b, europe-west4-a, us-east1-d,
us-central1-b, asia-northeast1-b or us-south1-a. Config is ready: `configs/v6e-8-qwen72b.env` (Qwen2.5-72B-Instruct, TP=8).

## Cost
About $1 of TPU time so far (covered by credits): ~0.3 chip-h on-demand v6e-1 ($0.81) plus ~0.1 chip-h of a preempted spot VM (~$0.10).
The full v6e-1 cache on + off sweep took ~7 minutes of benchmark time.

## Next
EAGLE3 speculative decoding for agent turns; host-memory KV offload for long sessions; longer sessions (20+ turns) where per-session history dominates.
