# DRAFT, not posted: #TorchTPUSprint post

Agent workloads are not chat workloads. Every turn resends the tool schemas plus a growing history,
so serving speed depends on prefix-cache reuse. But for hybrid linear-attention models the cache can hurt.

For #TorchTPUSprint I benchmarked SGLang (sgl_jax) on Cloud TPU v6e with a multi-turn agent workload
(11,000+ requests, 0 errors):

Qwen2.5-7B, 1x v6e chip
- Radix prefix cache at 32 agent sessions: 1,523 vs 930 output tok/s (+64%), TTFT p99 138 ms vs 598 ms.
  87% of prompt tokens served from cache; 100% of requests meet TTFT<200ms / ITL<25ms.
- Peak 2,755 tok/s at c=96 (~$0.10 per 1M output tokens at spot); past that the KV pool evicts prefixes and both throughput and hit rate fall.
- Decode runs at 55-73% of the HBM roofline.

Qwen3.8-27B (hybrid linear attention), v6e-4, TP=4
- Prefix cache made it slower at every concurrency: hits rounded to 1,024-token blocks (32-38%),
  and recurrent-state buffers cut max running requests 96 -> 27.
- Cache off: 968 tok/s at c=64; 12 ms ITL at c=1 (71% of roofline).

Whole study: ~$14 of TPU time.

Open harness (any OpenAI-compatible server): <repo link once public>
Write-up: <link>

#TorchTPUSprint #SGLang #TPU
