# DRAFT, not posted: SGLang on TPU v6e, multi-turn agent serving performance

**TL;DR (fill in after runs):** On a single v6e chip, Llama-3.1-8B served by SGLang holds TTFT at __ ms through turn 6
with prefix caching on, versus __ ms with it off. That is __x lower late-turn TTFT, at __ output tok/s with 32 concurrent agent sessions.

## Why agents are different
Agent sessions resend a long shared system prompt (tool schemas) and append a tool result every turn,
so the prompt grows each turn. Chat benchmarks (one-shot random prompts) miss this. The question is
whether the radix prefix cache keeps per-turn latency flat on TPU.

## Setup
- Hardware: Cloud TPU v6e-1 (8B) and v6e-8 (70B, TP=8), spot, us-east5.
- Server: SGLang TPU backend (sgl_jax, documented v6e path); TorchTPU container tried: __.
- Workload: N sessions x 6 turns, 600-word shared system prompt, 150-word tool results, 128 max output tokens.
  Concurrency 1/4/8/16/32. Harness: github.com/kitrakrev/sglang-tpu-agentbench (agentbench).
- Metrics: TTFT p50/p99, ITL p50/p99, output tok/s, median TTFT per turn, cached prompt tokens.

## Results
| model / chip | cache | c | tok/s | TTFT p50 | TTFT p99 | ITL p50 | TTFT turn0 -> turn5 |
|---|---|---|---|---|---|---|---|
| 8B / v6e-1 | on | 32 | | | | | |
| 8B / v6e-1 | off | 32 | | | | | |
| 70B / v6e-8 | on | 32 | | | | | |

(chart: TTFT by turn, cache on vs off; chart: tok/s vs concurrency)

## Findings
1. __
2. __
3. Gotchas: XLA compile on first requests (warmup needed; JAX_COMPILATION_CACHE_DIR), __

## Cost
Total TPU spend: $__ for __ chip-hours.

## Next
EAGLE3 speculative decoding for agent turns; host-memory KV offload for long sessions.
