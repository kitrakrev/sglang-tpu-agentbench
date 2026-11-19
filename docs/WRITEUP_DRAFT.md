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

## Run B (2026-11-18): v6e-4 Qwen3.8-27B-pi-adapted, TP=4 (on-demand, europe-west4-a, ~50 min VM time)
Model: `Qwen/Qwen3.8-27B` + `harness-hub/Qwen3.8-27B-pi-adapted` LoRA (r=32, alpha=64, 256 pairs) merged on the VM host
with `scripts/merge_lora.py` (0 unmatched keys). Hybrid arch: 48 Gated-DeltaNet (linear-attention) + 16 full-attention layers.
0 errors on 3,456 requests. Full table: `results/v6e-4-qwen38-27b-pi/metrics.md`; roofline: `roofline.md`.

| cache | c | out tok/s | tok/s/chip | TTFT p50/p99 ms | ITL p50/p99 ms | cached % | SLO % | goodput tok/s | $/1M out OD / spot | % roofline |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 1 | 78 | 19.6 | 102 / 138 | 12.0 / 12.0 | 0 | 100 | 78 | 38.3 / 14.2 | 71% |
| off | 8 | 430 | 107.6 | 173 / 506 | 15.9 / 27.2 | 0 | 52.6 | 222 | 6.97 / 2.58 | 54% |
| off | 16 | 655 | 163.6 | 210 / 872 | 21.1 / 47.1 | 0 | 19.5 | 127 | 4.58 / 1.70 | 41% |
| off | 32 | 893 | 223.2 | 255 / 1,450 | 32.3 / 62.7 | 0 | 3.0 | 24 | 3.36 / 1.24 | 28% |
| off | **64** | **968** | 242.0 | 534 / 3,060 | 55.8 / 169.6 | 0 | 0.1 | 1 | **3.10 / 1.15** | 17% |
| on | 1 | 36 | 9.0 | 117 / 244 | 26.7 / 26.8 | 38.0 | 0 | 0 | 83.5 / 30.9 | 32% |
| on | 4 | 237 | 59.3 | 183 / 521 | 14.5 / 19.5 | 31.9 | 55.2 | 133 | 12.6 / 4.68 | 59% |
| on | 16 | 536 | 133.9 | 265 / 1,915 | 23.7 / 54.6 | 31.5 | 14.8 | 77 | 5.60 / 2.07 | 37% |
| on | 32 | 520 | 129.9 | 1,588 / 6,167 | 46.6 / 98.5 | ~0 | 0 | 0 | 5.77 / 2.14 | 19% |
| on | 64 | 517 | 129.3 | 8,659 / 11,443 | 46.7 / 135.3 | ~0 | 0 | 0 | 5.80 / 2.15 | 20% |

Roofline assumes 27.8e9 params (55.6 GB incl. the unused vision tower, so slightly pessimistic) and full-attention KV only
(2x16x4x256x2 = 64 KiB/token); Gated-DeltaNet recurrent state (~38 MB/request) is not modelled, so % roofline at high c is understated.

Radix cache on a hybrid linear-attention model (the main finding):
1. sgl_jax refuses the plain radix cache for this arch: `Hybrid recurrent state models require --disable-radix-cache (legacy) or
   --enable-unified-radix-tree`. The unified tree then requires `--page-size 1` unless `--enable-recurrent-extra-buffer`.
   With mem-fraction 0.85 the extra buffer OOMs at compile (HLO temporaries 33.1 G > 31.2 G HBM/chip); 0.7 + chunked prefill 1024 works.
   Failed server logs are kept in `results/v6e-4-qwen38-27b-pi/logs/*FAILED.log` / `*OOM.log`.
2. Prefix caching helps far less than on a pure-attention model: hit rate 32-38% (vs 84-88% for Qwen2.5-7B) because a
   prefix is reusable only where a recurrent-state snapshot exists, so hits round down to 1,024-token boundaries (the shared
   system prompt) and per-session history is mostly recomputed. At c>=32 hits drop to ~0 as snapshot slots are evicted.
3. The cost of caching is concurrency: recurrent snapshots cut the scheduler to max_num_reqs=27 (vs 96 with cache off), so
   throughput plateaus at ~520 tok/s from c=16 and TTFT p99 explodes (6-11 s) from queueing. Cache off reaches 968 tok/s at c=64.
4. Net: on this model **cache off wins at every concurrency** (1.1x at c=4, 1.9x at c=64). c=1 cache-on ITL (26.7 ms vs 12.0)
   points to extra snapshot/copy work per decode step. For agent serving of hybrid models the radix cache needs a cheaper
   recurrent-state checkpoint scheme before it pays off.
5. Startup 340-391 s (XLA compile of TP=4 + Pallas kernels; tuned block sizes missing for head_dim 256 / page 128, heuristic used).
   Spot v6e-4 had no capacity in 10+ zones; this ran on-demand ($2.70/chip-h).

## GPU baseline (2026-11-18): Qwen2.5-7B on Modal A100-80GB and H100 SXM
Same workload as the v6e-1 runs (agentbench, seed 42, 4 sessions/level x 6 turns, 600-word system prompt, 150-word tool
results, 128 max out tokens), same goodput SLO and metrics. Server: upstream SGLang v0.5.2 (`lmsysorg/sglang:v0.5.2-cu126`),
`--mem-fraction-static 0.85 --chunked-prefill-size 2048 --max-running-requests 128 --enable-cache-report`, page size 1
(GPU default; TPU used 128), CUDA graphs on, cache off = `--disable-radix-cache`. Client runs in the same Modal container
against localhost. GPUs: `NVIDIA A100-SXM4-80GB` and `NVIDIA H100 80GB HBM3` (pinned with `gpu="H100!"` so Modal cannot
upgrade it to H200). 0 errors on 11,520 requests. Script: `modal/gpu_bench.py`; tables: `results/{a100,h100}-qwen7b/metrics.md`, `roofline.md`.

Prices (Modal on-demand, modal.com/pricing, 2026-11-18): A100-80GB $0.000694/s = **$2.50/h**, H100 $0.001097/s = **$3.95/h**;
v6e-1 $2.70/chip-h on-demand. Roofline bandwidth: A100 80GB SXM 2.039 TB/s, H100 SXM 3.35 TB/s (NVIDIA A100 / H100 datasheets;
Modal docs confirm its H100s are SXM; nvidia-smi confirms A100 SXM4), v6e 1.64 TB/s. % roofline = HBM floor ITL / measured ITL p50.

### TPU v6e-1 vs GPU, radix cache on
TPU c=64/96 come from Run A (`hiconc`, same settings, separate VM). No TPU cache-off row exists above c=32.

| c | device | out tok/s | TTFT p50 / p99 ms | ITL p50 / p99 ms | cached % | SLO % | goodput tok/s | $/1M out (OD) | % roofline |
|---|---|---|---|---|---|---|---|---|---|
| 1 | v6e-1 | 75 | 30 / 33 | 12.9 / 12.9 | 84.1 | 100 | 75 | 10.01 | 73% |
| 1 | A100 | 90 | 41 / 51 | 10.3 / 10.3 | 88.6 | 100 | 90 | 7.69 | 73% |
| 1 | H100 | **149** | **28 / 33** | **6.2 / 6.3** | 88.7 | 100 | 149 | **7.34** | 74% |
| 8 | v6e-1 | 515 | 40 / 65 | 14.5 / 15.4 | 87.7 | 100 | 515 | 1.46 | 67% |
| 8 | A100 | 594 | 52 / 107 | 12.3 / 14.3 | 92.3 | 100 | 594 | 1.17 | 63% |
| 8 | H100 | **962** | 31 / 76 | **7.5 / 9.2** | 92.1 | 100 | 962 | **1.14** | 63% |
| 16 | v6e-1 | 913 | 41 / 99 | 16.4 / 18.4 | 87.2 | 100 | 913 | 0.82 | 61% |
| 16 | A100 | 985 | 67 / 153 | 14.6 / 17.0 | 92.0 | 100 | 985 | **0.71** | 55% |
| 16 | H100 | **1,592** | 44 / 102 | **9.1 / 11.0** | 92.3 | 100 | 1,592 | **0.69** | 54% |
| 32 | v6e-1 | 1,515 | 42 / **134** | 19.8 / 22.5 | 87.6 | **100** | 1,515 | 0.50 | **55%** |
| 32 | A100 | 1,548 | 89 / 410 | 18.9 / 22.8 | 92.0 | 96.7 | 1,522 | 0.45 | 46% |
| 32 | H100 | **2,502** | 68 / 529 | **11.1 / 14.9** | 92.6 | 96.4 | **2,457** | **0.44** | 48% |
| 64 | v6e-1 | 2,084 | 65 / **422** | 26.6 / 32.9 | 83.7 | 26.4 | 616 | 0.36 | 47% |
| 64 | A100 | 2,339 | 150 / 452 | 24.2 / 30.5 | 92.6 | 36.4 | 873 | 0.30 | 41% |
| 64 | H100 | **4,249** | 152 / 545 | **12.3 / 16.9** | 93.0 | **81.4** | **3,679** | **0.26** | 50% |
| 96 | v6e-1 | 2,755 | **70 / 520** | 30.1 / 35.7 | 87.9 | 7.4 | 301 | 0.27 | 46% |
| 96 | A100 | 3,001 | 310 / 821 | 26.3 / 34.0 | 94.0 | 5.3 | 124 | 0.23 | 43% |
| 96 | H100 | **5,197** | 238 / 618 | **13.2 / 19.7** | 94.3 | 31.5 | 1,747 | **0.21** | 52% |

### Cache off (c=1..32 where TPU has data)
| c | v6e-1 tok/s / ITL p50 / TTFT p99 | A100 | H100 |
|---|---|---|---|
| 1 | 72 / 12.9 / 81 | 83 / 10.3 / 160 | 144 / 6.3 / 53 |
| 8 | 433 / 16.9 / 187 | 398 / 17.3 / 382 | 781 / 9.1 / 151 |
| 16 | 668 / 22.3 / 341 | 532 / 26.5 / 722 | 1,134 / 12.5 / 298 |
| 32 | 930 / 32.4 / 598 | 653 / 42.9 / 1,207 | 1,424 / 19.6 / 743 |
| 64 | - | 732 / 76.3 / 2,380 | 1,663 / 32.4 / 1,017 |
| 96 | - | 754 / 109.0 / 3,784 | 1,743 / 44.9 / 1,734 |

Findings (honest read):
1. **H100 wins outright** on every throughput and ITL row: 1.65-2.0x v6e-1 tok/s at every concurrency (5,197 vs 2,755 at c=96),
   ITL ~half, and the lowest $/1M output at every c despite costing 1.46x per hour (e.g. $0.21 vs $0.27 at c=96, $0.44 vs $0.50 at c=32).
   It is the only device that holds the goodput SLO for most requests at c=64 (81% vs 26% v6e, 36% A100).
2. **A100-80GB vs v6e-1 is close to a tie** with cache on: A100 has 2-20% more tok/s, slightly lower ITL and 10-23% lower $/1M
   (cheaper per hour on Modal, $2.50 vs $2.70), but v6e-1 has clearly better tail TTFT: 134 vs 410 ms p99 at c=32, and the TPU is
   the only device at 100% SLO at c=32. With cache off, v6e-1 beats A100 from c=8 up (930 vs 653 tok/s at c=32, ITL 32 vs 43 ms).
3. Per HBM byte, all three land in the same place: 73-74% of the decode roofline at c=1, 41-55% at c>=32. v6e is ~55% at c=32 vs
   46-48% for the GPUs. The H100 gap is bandwidth (3.35 vs 1.64 TB/s = 2.0x) more than software efficiency.
4. Radix cache matters on GPU as much as on TPU: cache on gives 2.4x (A100) and 1.8x (H100) tok/s at c=32, and more at c>=64.
   GPU hit rates are a bit higher (92-94% vs 84-88%) because page size 1 matches prefixes to the token; TPU page 128 rounds down.
5. GPU TTFT p99 at c=32 (410-529 ms) is worse than TPU's 134 ms; likely because the GPU scheduler admits the burst of new-session prefills together
   (not profiled). Neither server was tuned per platform beyond the shared flags.
6. Caveats: SGLang GPU (v0.5.2, mature CUDA kernels + CUDA graphs) vs sgl_jax (young); different page size; Modal per-second
   billing vs GCE hourly; TPU c=64/96 from a different VM than c<=32. Startup: A100 171 s / H100 76 s first (cold image + weight load), ~41-46 s second.

Charts: `results/{a100,h100}-qwen7b/charts/` (same set as v6e-1 plus `roofline_on.png` / `roofline_off.png`).

## Run C (2026-11-19): OLMo 2 / Olmo 3 ports and int8 weights, v6e-1
On-demand v6e-1 in asia-northeast1-b (~2 h VM time, ~$5.3). A first spot attempt in europe-west4-a was preempted after
~23 min (maintenance event -> PREEMPTED) and its partial results were lost. sglang-jax from `kitrakrev/sglang-jax`
branch `add-olmo3-support` (upstream main + OLMo 2 + Olmo 3 ports). 0 errors on 4,800 requests.
Tables: `results/v6e-1-olmo2-7b/metrics.md`, `results/v6e-1-olmo3-7b/metrics.md`, `results/v6e-1-qwen7b-int8/metrics.md`.

### int8 weight-only quantization, Qwen2.5-7B, cache on
Built-in `--quantization-config-path=int8.yaml` (per-channel int8 weights for every linear layer, bf16 activations and KV,
quantized at load from the bf16 checkpoint). Baseline re-run on the same VM and build.

| c | bf16 tok/s | int8 tok/s | bf16 ITL p50 ms | int8 ITL p50 ms | bf16 TTFT p99 | int8 TTFT p99 | SLO % bf16 / int8 |
|---|---|---|---|---|---|---|---|
| 1 | 74 | **135** | 13.0 | **7.1** | 33 | 27 | 100 / 100 |
| 4 | 275 | **487** | 13.7 | **7.7** | 52 | 42 | 100 / 100 |
| 8 | 506 | **875** | 14.7 | **8.5** | 64 | 43 | 100 / 100 |
| 16 | 901 | **1,502** | 16.6 | **9.9** | 112 | 78 | 100 / 100 |
| 32 | 1,505 | **2,342** | 20.1 | **12.9** | 203 | 249 | 98.3 / 98.6 |

1. int8 weights make decode **1.84x faster at c=1** (ITL 13.0 -> 7.1 ms) and give **+56% tok/s at c=32** (2,342 vs 1,505).
   That matches decode being HBM-bound: weight bytes halve (15.2 -> ~7.6 GB) and the floor drops from 9.3 to ~4.6 ms.
2. KV pool grows from 12.4 to 18.4 GB (232k -> 345k tokens), so more sessions fit before prefix eviction.
3. $/1M output at c=32 on-demand: ~$0.32 vs $0.50 bf16.
4. **Not measured: output quality.** No accuracy eval was run on the int8 model; treat this as a speed result only.

### OLMo 2 7B and Olmo 3 7B (new sgl_jax ports)
Ports: `Olmo2ForCausalLM` (post-sublayer RMSNorm, full-width QK-norm) and `Olmo3ForCausalLM` (same block + 3:1
sliding-window attention, window 4096, YaRN RoPE). CPU parity vs HF transformers (tiny random fp32 models):
OLMo 2 max |prompt logprob diff| 2.9e-6, Olmo 3 1.9e-6 (window 3 so it really masks, YaRN on), identical greedy tokens.
On TPU both served the full agent workload with 0 errors; Olmo 3 used sgl_jax's hybrid SWA KV pool
(full layers 31,104 tokens, SWA layers 24,832).

| model | cache | c=1 tok/s | c=1 TTFT p50 ms | c=1 ITL p50 ms | c=32 tok/s | c=32 TTFT p50 ms | cached % |
|---|---|---|---|---|---|---|---|
| OLMo-2-1124-7B-Instruct | on | 66 | 219 | 13.4 | 162 | 1,374 | 79 -> 44 |
| OLMo-2-1124-7B-Instruct | off | 45 | 938 | 13.5 | 98 | 15,358 | 0 |
| Olmo-3-7B-Instruct | on | 65 | 124 | 13.4 | 65 | 15,881 | 84 -> 54 |
| Olmo-3-7B-Instruct | off | 40 | 723 | 13.4 | 38 | 28,857 | 0 |
| Qwen2.5-7B-Instruct (ref) | on | 74 | 30 | 13.0 | 1,505 | 42 | 88 |

Decode is fine (ITL 13.4 ms at c=1, same as Qwen2.5-7B, ~70% of the HBM roofline), but **prefill is 10-30x slower** and
throughput collapses with concurrency. Root cause from the server logs: OLMo 2/3 7B use full multi-head attention
(32 KV heads, vs 4 for Qwen2.5-7B). sgl_jax's ragged paged attention v3 kernel has no tuned block sizes for that shape on
v6e (`RPA v3 tuned-block-size LOOKUP MISS ... 32, 32, 128`), and the VMEM heuristic picks **bq=2, bkv=512** for
prefill/mixed batches, vs bq=32, bkv=1024 for Qwen's 4 KV heads. Each prefill chunk then takes 1-2 s, so requests queue
(the scheduler shows 2 running, KV usage ~7%). MHA also makes KV 8x bigger per token (512 KiB vs 56 KiB), which shrinks
the pool to ~25-30k tokens. These are kernel-tuning issues, not port correctness: next step is adding v6e block sizes for
(32 q heads, 32 kv heads, head_dim 128) to `TUNED_BLOCK_SIZES_V3`, then re-running.

## v6e-8, Qwen2.5-72B
Not run: no v6e-8 capacity. Attempts:
- 2026-11-18 ~09:50-10:05 PST, spot: us-east5-b, europe-west4-a, us-east1-d, us-central1-b, asia-northeast1-b, us-south1-a.
- 2026-11-19 ~03:45-03:58 UTC, spot (async create in all 6 zones; every request ended without a VM) and on-demand:
  europe-west4-a, us-east5-b, asia-northeast1-b "no more capacity in the zone"; us-south1-a "Insufficient capacity";
  us-central1-b and us-east1-d "User does not have permission to submit requests for accelerator type v6e-8" (no v6e-8 quota there).
Config is ready: `configs/v6e-8-qwen72b.env` (Qwen2.5-72B-Instruct, TP=8). bf16 72B (~145 GB) does not fit a v6e-4 (128 GB HBM).

## Cost
2026-11-19 Run C: v6e-1 spot ~0.4 h (preempted, ~$0.4) + v6e-1 on-demand ~1.97 h x $2.70 = ~$5.3. v6e-8 attempts: no VM was created, $0.
GPU baseline (Modal, 2026-11-18): A100-80GB ~32 min container time incl. smoke test (~$1.35) + H100 ~9 min (~$0.60) + CPU download/volume (<$0.05) = **~$2.0** upper bound (app wall times; GPU billed only while containers run).
2026-11-18 Runs A+B: v6e-1 spot ~0.45 h (~$0.5-1.2) + v6e-4 on-demand ~0.85 h x 4 chips x $2.70 = ~$9.2. Earlier: ~$1 of TPU time so far (covered by credits): ~0.3 chip-h on-demand v6e-1 ($0.81) plus ~0.1 chip-h of a preempted spot VM (~$0.10).
The full v6e-1 cache on + off sweep took ~7 minutes of benchmark time.

## Next
EAGLE3 speculative decoding for agent turns; host-memory KV offload for long sessions; longer sessions (20+ turns) where per-session history dominates.
