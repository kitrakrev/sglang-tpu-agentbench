# DRAFT, not posted: #TorchTPUSprint post

Agent workloads are not chat workloads. Every turn resends the tool schemas plus a growing history,
so serving speed depends on prefix-cache reuse.

For #TorchTPUSprint I benchmarked SGLang (sgl_jax) on Cloud TPU v6e with a multi-turn agent workload:
- Qwen2.5-7B on one v6e chip, 32 concurrent agent sessions: radix prefix cache gives 1,515 vs 930 output tok/s (+63%)
  and TTFT p99 134 ms vs 598 ms (4.5x lower). 84-88% of prompt tokens come from cache.
- Median TTFT stays flat at ~40 ms from turn 1 to turn 6 with the cache on.
- About $0.50 per 1M output tokens at on-demand price (~$0.18 at spot).
- Qwen2.5-72B on v6e-8: <pending, no spot capacity yet>
- Whole study so far cost about $1 in TPU time

Open harness (works on any OpenAI-compatible server): <repo link once public>
Write-up: <link>

Thanks @Google Cloud TPU team and the SGLang folks. #TorchTPUSprint #SGLang #TPU
