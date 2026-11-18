# Next experiments (ranked by expected gain per $)

Derived from [the TPU optimization guide](https://ai-hypercomputer.github.io/tpu-codesign/) and docs/ROOFLINE.md. Nothing has been launched. Costs assume v6e-1 on-demand at about $2.70/chip-h. The current full sweep takes about 7 min.

| # | Experiment | Why (guide) | Expected effect | Est. cost |
|---|---|---|---|---|
| 1 | Concurrency c=64, 96, 128 (cache on), with `--max-running-requests 128` | Decode is memory-bound until B is about 500. Amortize the weight read ([arch 8](https://ai-hypercomputer.github.io/tpu-codesign/architecture-choices/)) | tok/s about 2,400-2,900 at c=128 if ITL stays under 45 ms. The KV pool (about 170 seqs) sets the ceiling. Watch TTFT p99 / goodput. | ~10 min, ~$0.50 |
| 2 | Page size 128 vs 256 | The guide suggests KV block 256 and tuning pages separately from attention blocks ([sys 12](https://ai-hypercomputer.github.io/tpu-codesign/systems-mapping/)) | 5-15% lower ITL at c>=16 if the attention kernel is the gap. Small prefix-cache granularity loss. | ~10 min, ~$0.50 |
| 3 | Chunked prefill 512/1024/2048/4096 at c=32 | Tune chunk size together with batching ([appendix B](https://ai-hypercomputer.github.io/tpu-codesign/appendix-b/)) | Smaller chunks should lower ITL p99 and raise TTFT. Pick by goodput (TTFT<200 ms, TPOT<25 ms). | ~15 min, ~$0.70 |
| 4 | int8 weight-only quantization (if sgl_jax supports it) | Fewer bytes per token ([arch 6/8](https://ai-hypercomputer.github.io/tpu-codesign/architecture-choices/)) | Floor 9.3 to about 4.7 ms. Expect ITL about 1.5x lower at c<=8. Needs a quality check. | ~15 min, ~$0.70 |
| 5 | int8 KV cache | The guide suggests FP8 KV; v6e has no FP8, so use int8 | About 2x KV capacity (about 340 seqs), and lowers the KV term at high c | ~10 min, ~$0.50 |
| 6 | XProf trace of a decode step at c=1 and c=32 | Measure, don't guess ([diagnostics](https://ai-hypercomputer.github.io/tpu-codesign/blueprint-and-diagnostics/)) | Confirms which gap hypothesis is right | ~5 min, ~$0.25 |
| 7 | EAGLE3 / speculative decoding at c=1-8 | One weight read verifies several tokens ([arch 8.2](https://ai-hypercomputer.github.io/tpu-codesign/architecture-choices/)) | 1.5-2.5x tokens per step at low c. The gain shrinks as c grows. | ~20 min, ~$1 |
| 8 | v6e-8: 8x DP replicas of 7B vs TP=8 (and 72B TP=8 when spot capacity returns) | Compare wider TP against more replicas at a fixed chip count ([appendix B](https://ai-hypercomputer.github.io/tpu-codesign/appendix-b/)) | DP should win by about 8x on throughput for 7B. TP lowers latency only. | ~$5-10 per hour on spot |

Experiments 1-3 plus 6 fit in one VM session (about 40 min, about $2).

## Candidate: agent-tuned 27B model (added 2026-11-18)

- Model: [`harness-hub/Qwen3.8-27B-pi-adapted`](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) — Qwen3.8-27B + RL/LoRA adapter for the pi coding agent (SWE-bench Pro Hard 51: 29.4% → 62.7%; Terminal-Bench 2.1: 39.5% → 55.8%, per model card/announcement).
- Why: real multi-turn coding-agent model = exactly this benchmark's workload; stronger story than generic chat model.
- Memory: ~54 GB bf16 weights → does not fit v6e-1 (32 GB HBM). Options: v6e-4 TP=4 (~13.5 GB/chip, room for KV) or int8 weights on v6e-2.
- Checks first: sgl_jax support for Qwen3 arch (+ LoRA merge — merge adapter offline into base weights before serving); licence/ungated download.
- Cost est.: v6e-4 on-demand ~$10.80/h → ~1 h sweep (cache on/off, c=1..64) ≈ $11; spot ≈ $4 if capacity.
