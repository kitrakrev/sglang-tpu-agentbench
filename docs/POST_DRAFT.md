# DRAFT, not posted: #TorchTPUSprint post

Agent workloads are not chat workloads. Every turn resends the tool schemas plus a growing history,
so serving speed depends on prefix-cache reuse.

For #TorchTPUSprint I benchmarked SGLang on Cloud TPU v6e with a multi-turn agent workload:
- Llama-3.1-8B on v6e-1: prefix cache keeps turn-6 TTFT at __ ms (vs __ ms without), __ tok/s at 32 sessions
- Llama-3.1-70B on v6e-8: __
- Whole study cost $__ in TPU time

Open harness (works on any OpenAI-compatible server): <repo link once public>
Write-up: <link>

Thanks @Google Cloud TPU team and the SGLang folks. #TorchTPUSprint #SGLang #TPU
