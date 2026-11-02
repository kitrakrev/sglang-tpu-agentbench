# sglang-tpu-agentbench

Benchmark for **multi-turn agent serving** with SGLang on Cloud TPU (TorchTPU Sprint 2026).

Agent workloads are not chat workloads. Each session shares a long system prompt
(tool schemas), and every turn appends a tool result, so context grows turn by
turn. Prefix caching decides whether latency stays flat or climbs. This harness
measures that.

## What it measures
- TTFT p50/p90/p99, inter-token latency, output tokens/s, requests/s
- Median TTFT **per turn** (does latency grow as context grows?)
- Cached prompt tokens reported by the server (prefix-cache reuse)

## Quick start (no TPU needed)
```bash
uv venv && uv pip install -e '.[dev]'
pytest -q
python -m agentbench.mock_server --port 30000 &
python -m agentbench.run --url http://localhost:30000 --sessions 32 --turns 6 --concurrency 8
```
The mock server fakes a prefix cache so the harness can be tested locally.

## On TPU
```bash
scripts/tpu_up.sh v6e-1            # dev: 8B model
scripts/serve_and_bench.sh         # run on the VM; concurrency sweep 1..32
scripts/idle_delete.sh             # deletes the VM after 2h idle
```
Plan: develop with an 8B model on v6e-1, final numbers with a 70B model on v6e-8.

## First real run (Colab T4, SGLang 0.5.23, Qwen2.5-0.5B, 6-turn sessions, 64 max tokens)
| concurrency | output tok/s | TTFT p50 |
|---|---|---|
| 1 | 98 | 86 ms |
| 4 | 214 | 126 ms |
| 8 | 88 | 478 ms |
| 16 | 150 | 314 ms |

Smoke test only: the c=8 run was the first after server start (no warmup, 21 s p99 TTFT outlier from kernel compile), and cache reporting was off. Both fixed since. Not TPU numbers.

## Status
- [x] Harness, mock server, tests
- [x] Real SGLang server end-to-end (Colab T4)
- [ ] Confirm SGLang TPU launch command (`SERVER_CMD`)
- [ ] v6e-1 8B baseline, prefix cache on vs off
- [ ] v6e-8 70B results + write-up
